#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:21:23 2026

@author: mollyrobertson

"""

import numpy as np
import scipy.sparse as sp
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time



def optimiser(
    routes, trucks, times, route_truck, active_routes_at, route_energy, route_diesel_used, route_departure, route_return, overlapping_routes,
    pv_profiles_t, pv_min, spot_price_t, grid_price_t, feed_in_tariff_t, whole_market, month_limits, months, df_data, dict_limits, network_access_cost,
    diesel_price_t,

    duration_days, dt, years, first_t, last_t, batteries, chargers,
    n_trucks,

    sb_cap, sb_min_soc, sb_max_soc, sb_initial_soc, sb_charge_efficiency, sb_cap_min,
    sb_discharge_efficiency, sb_self_discharge, sb_max_charge_power,

    battery_cost, charger_power_kw, charger_cost, charger_min,

    bess_initial_soc, bess_min_soc, bess_max_soc, bess_max_cap, bess_min_cap,
    bess_max_charge_power, bess_max_discharge_power,
    bess_charge_efficiency, bess_discharge_efficiency, bess_self_discharge,
    bess_charge_c_rate, bess_discharge_c_rate,

    max_grid_import, max_grid_export,
    existing_grid_capacity, max_grid_capacity,

    pv_efficiency, pv_max_kw, pv_capex_per_kw,

    bess_capex_per_kwh, retrofit_cost, diesel_capex,

    discount_rate, project_lifetime,

    sb_lifetime, charger_lifetime, bess_lifetime, pv_lifetime, retrofit_lifetime, diesel_lifetime, 

    sb_om, charger_om, bess_om, pv_om, retrofit_om, diesel_om,

    route_year,             # y(r)
    trucks_in_year,         # I(y)
    fleet_size,             # N_fleet(y)
    truck_first_year,       # y_first(i)
    truck_base_electric,    # z_base(i)

    routes_departing_at,    # D(t)
    route_pre_departure,    # t_pre(r)

    aux_load_t,             # P_aux(t)

    time_limit=None,        # demo-mode solver time cap (seconds); None = no cap
    mip_gap=0.05,            # demo-mode MIP gap; tighten for full-accuracy runs,
):



    # ============================================================
    # Tight Implied Bounds
    # ============================================================
    # Give Gurobi tighter
    # limits up front for presolve and LP conditioning.
    bess_charge_bound = min(bess_max_charge_power, bess_charge_c_rate * bess_max_cap)
    bess_discharge_bound = min(bess_max_discharge_power, bess_discharge_c_rate * bess_max_cap)
    grid_import_bound = min(max_grid_import, max_grid_capacity)
    grid_export_bound = min(max_grid_export, max_grid_capacity)


    # ============================================================
    # Model
    # ============================================================

    m = gp.Model("battery_selection")

    # Presolve left at default - Presolve=2 was tested and found slower
    # on this model (see modelling-learnings.md).
    m.setParam("MIPGap", mip_gap)
    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)

    # Restored from the team's empirical solver-tuning comparison runs.
    m.setParam("MIPFocus", 3)
    m.setParam("Method", 2)
    m.setParam("Symmetry", 2)
    m.setParam("Heuristics", 0.2)

    # Cuts=1 (down from 2) + CutPasses cap: logs showed Cuts=2 driving
    # 715 Relax-and-lift rounds for a shrinking bound improvement, and
    # NodeCount=1 on every run (the model never actually branches).
    # This forces the solver to stop cutting sooner and try branching.
    m.setParam("Cuts", 1)
    m.setParam("CutPasses", 10)



    # ============================================================
    # Index maps and sparse structure matrices (for matrix constraints)
    # ============================================================

    battery_idx = {b: idx for idx, b in enumerate(batteries)}
    route_idx   = {r: idx for idx, r in enumerate(routes)}
    time_idx    = {t: idx for idx, t in enumerate(times)}

    NB, NR, NT = len(batteries), len(routes), len(times)

    # Active[t,r] = 1 if route r is underway at timestep t     (C2)
    _Active = sp.lil_matrix((NT, NR))
    for t in times:
        for r in active_routes_at[t]:
            _Active[time_idx[t], route_idx[r]] = 1
    Active = _Active.tocsr()

    # DepartWeighted[t,r] = route_energy[r] if r departs at t   (SOC2)
    _DepartWeighted = sp.lil_matrix((NT, NR))
    for t in times:
        for r in routes_departing_at[t]:
            _DepartWeighted[time_idx[t], route_idx[r]] = route_energy[r]
    DepartWeighted = _DepartWeighted.tocsr()

    # Pre[r,t] = 1 if t is route r's pre-departure timestep     (SOC3)
    _Pre = sp.lil_matrix((NR, NT))
    for r in routes:
        _Pre[route_idx[r], time_idx[route_pre_departure[r]]] = 1
    Pre = _Pre.tocsr()

    # Diagonal route-energy matrix - avoids ndarray * MVar elementwise
    # multiply (unsupported dispatch); fold constants into A @ x form.
    route_energy_vec = np.array([route_energy[r] for r in routes])
    RE_diag = sp.diags(route_energy_vec)

    first_row = time_idx[first_t]
    last_row = time_idx[last_t]



    # ============================================================
    # Decision Variables
    # ============================================================

    N_EV = m.addVars(years, vtype=GRB.INTEGER, lb=0, name="N_EV")
    N_D = m.addVars(years, vtype=GRB.INTEGER, lb=0, name="N_D")
    N_SB = m.addVar(vtype=GRB.INTEGER, lb=sb_cap_min, ub=len(batteries), name="N_SB")   #Change
    N_CS = m.addVar(vtype=GRB.INTEGER, lb=charger_min, ub=len(chargers), name="N_CS")   #Change

    z = m.addVars(trucks, years, vtype=GRB.BINARY, name="z")
    k = m.addVars(trucks, years, vtype=GRB.BINARY, name="k")

    # Matrix-form variables (SB/charging/SOC/E1 blocks below use these)
    w       = m.addMVar(NB,       vtype=GRB.BINARY,    name="w")
    a       = m.addMVar((NR, NB), vtype=GRB.BINARY,    name="a")
    x_SB_ch = m.addMVar((NT, NB), vtype=GRB.BINARY,    name="x_SB_ch")
    E_SB    = m.addMVar((NT, NB), lb=0, ub=sb_max_soc * sb_cap, name="E_SB")
    P_SB_ch = m.addMVar((NT, NB), lb=0, ub=sb_max_charge_power, name="P_SB_ch")

    Q_BESS = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bess_max_cap, name="Q_BESS")
    P_BESS_ch_inst = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bess_max_charge_power, name="P_BESS_ch_inst")
    P_BESS_dis_inst = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bess_max_discharge_power, name="P_BESS_dis_inst")
    E_BESS = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, ub=bess_max_soc * bess_max_cap, name="E_BESS")
    P_BESS_ch = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, ub=bess_charge_bound, name="P_BESS_ch")
    P_BESS_dis = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, ub=bess_discharge_bound, name="P_BESS_dis")
    x_BESS = m.addVars(times, vtype=GRB.BINARY, name="x_BESS")

    E_PV_cap = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=pv_max_kw, name="E_PV_cap")
    P_PV_generated = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_PV_generated")

    S_G = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_grid_capacity, name="S_G")
    P_G_imp = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, ub=grid_import_bound, name="P_G_imp")
    P_G_exp = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, ub=grid_export_bound, name="P_G_exp")
    x_G = m.addVars(times, vtype=GRB.BINARY, name="x_G")

    # ============================================================
    # 3. Fleet Conversion Constraints
    # ============================================================

    # Previously converted trucks, must remain converted
    for i in trucks:
        if truck_base_electric[i] == 1:
            m.addConstr(z[i, years[0]] == 1, name=f"F0_BaseElectric[{i}]")

    # F1. Number of Electric Trucks
    for y in years: m.addConstr(N_EV[y] == gp.quicksum(z[i, y] for i in trucks_in_year[y]), name=f"F1_NumberElectricTrucks[{y}]")

    # F2. Fleet-Size Conservation
    for y in years: m.addConstr(N_D[y] + N_EV[y] == fleet_size[y], name=f"F2_FleetSizeConservation[{y}]")

    # F3. Irreversible Conversion
    for i in trucks:
        for y_idx in range(1, len(years)):
            y, y_prev = years[y_idx], years[y_idx - 1]
            if i in trucks_in_year[y] and i in trucks_in_year[y_prev]: m.addConstr(z[i, y] >= z[i, y_prev], name=f"F3_IrreversibleConversion[{i},{y}]")

    # F4. Conversion Event Indicator
    for i in trucks:
        first_year = truck_first_year[i]
        m.addConstr(k[i, first_year] == z[i, first_year] - truck_base_electric[i], name=f"F4_InitialConversion[{i},{first_year}]")
        for y_idx in range(1, len(years)):
            y, y_prev = years[y_idx], years[y_idx - 1]
            if y != first_year and i in trucks_in_year[y] and i in trucks_in_year[y_prev]: m.addConstr(k[i, y] == z[i, y] - z[i, y_prev], name=f"F4_ConversionEvent[{i},{y}]")

    # ============================================================
    # 5. Swappable Battery Inventory and Route Assignment (matrix)
    # ============================================================

    # SB1. Number of Swappable Batteries Purchased
    m.addConstr(N_SB == gp.quicksum(w[b] for b in batteries), name="SB1_NumberSwappableBatteries")

    # SB1A. Symmetry Breaking for Identical Candidate Batteries
    # Candidate batteries all share the same capacity, cost, efficiency
    # and operating constraints. Forcing lower-index batteries to be
    # purchased first removes duplicate-labelled solutions without
    # excluding any distinct physical design - the optimum is unchanged.
    m.addConstr(w[:-1] >= w[1:], name="SB1A_BatterySymmetry")

    # SB2. One Battery per Electric Route
    z_r = gp.MVar.fromlist([z[route_truck[r], route_year[r]] for r in routes])   # (NR,)
    m.addConstr(a.sum(axis=1) == z_r, name="SB2_OneBatteryPerElectricRoute")

    # SB3. Only Purchased Batteries May Be Assigned
    m.addConstr(a <= w, name="SB3_OnlyPurchasedBattery")   # broadcasts w (NB,) across NR rows

    # SB4 removed - implied by C2 below (C2 with x_SB_ch fixed at 0
    # reduces to SB4, so it was a redundant row).

    # ============================================================
    # 6. Swappable Battery Charging and Charger Assignment (matrix)
    # ============================================================

    # C1 removed - implied by C2 below (C2 with the active-route sum
    # fixed at 0 reduces to C1, so it was a redundant row).

    # C2. Battery cannot charge while on a route (also implies SB4 and C1)
    m.addConstr(x_SB_ch + Active @ a <= w, name="C2_NoChargeWhileRoute")

    # C3. Battery charging power limit
    m.addConstr(P_SB_ch <= sb_max_charge_power * x_SB_ch, name="C3_BatteryChargePower")

    # C4. Number of batteries charging cannot exceed number of chargers
    N_CS_vec = gp.MVar.fromlist([N_CS] * NT)
    m.addConstr(x_SB_ch.sum(axis=1) <= N_CS_vec, name="C4_ChargerCount")

    # C5. Total charging power cannot exceed installed charger power
    m.addConstr(P_SB_ch.sum(axis=1) <= charger_power_kw * N_CS_vec, name="C5_TotalChargerPower")

    # ============================================================
    # 7. Swappable Battery State of Energy (matrix)
    # ============================================================

    # SOC1. Stored-Energy Bounds
    m.addConstr(E_SB >= sb_min_soc * sb_cap * w, name="SOC1_SB_Min")
    m.addConstr(E_SB <= sb_max_soc * sb_cap * w, name="SOC1_SB_Max")

    # SOC2. Swappable-Battery State-of-Energy Balance
    departure_term = DepartWeighted @ a   # (NT, NB)
    m.addConstr(
        E_SB[1:, :] == (1 - sb_self_discharge) * E_SB[:-1, :]
                     + sb_charge_efficiency * dt * P_SB_ch[1:, :]
                     - departure_term[1:, :],
        name="SOC2_SB_EnergyBalance"
    )

    # SOC3. Sufficient Energy Before Departure
    route_term = RE_diag @ a   # (NR, NB), canonical A @ x form
    m.addConstr(
        Pre @ E_SB >= sb_min_soc * sb_cap * w + route_term,
        name="SOC3_SufficientDepartureEnergy"
    )

    # SOC4. Initial Swappable-Battery Energy
    m.addConstr(E_SB[first_row, :] == sb_initial_soc * sb_cap * w, name="SOC4_InitialSBEnergy")

    # SOC5. Terminal Swappable-Battery Condition
    m.addConstr(E_SB[last_row, :] >= E_SB[first_row, :], name="SOC5_TerminalSBEnergy")

    # ============================================================
    # 8. Stationary BESS Constraints
    # ============================================================

    # B1. Installed BESS Energy Capacity
    m.addConstr(Q_BESS <= bess_max_cap, name="B1_BESSCapacity")
    m.addConstr(Q_BESS >= bess_min_cap, name="BESS_Installed") #Change

    # B2. Installed BESS Power Capacities
    m.addConstr(P_BESS_ch_inst <= bess_max_charge_power, name="B2_BESSChargeCapacity")
    m.addConstr(P_BESS_dis_inst <= bess_max_discharge_power, name="B2_BESSDischargeCapacity")

    # B3. BESS Stored-Energy Bounds
    for t in times:
        m.addConstr(E_BESS[t] >= bess_min_soc * Q_BESS, name=f"B3_BESS_SOCMin[{t}]")
        m.addConstr(E_BESS[t] <= bess_max_soc * Q_BESS, name=f"B3_BESS_SOCMax[{t}]")

    # B4. BESS Charging-Power Limits
    for t in times:
        m.addConstr(P_BESS_ch[t] <= P_BESS_ch_inst, name=f"B4_InstalledChargePower[{t}]")
        m.addConstr(P_BESS_ch[t] <= bess_charge_bound * x_BESS[t], name=f"B4_ChargeMode[{t}]")
        m.addConstr(P_BESS_ch[t] <= bess_charge_c_rate * Q_BESS, name=f"B4_ChargeCRate[{t}]")

    # B5. BESS Discharging-Power Limits
    for t in times:
        m.addConstr(P_BESS_dis[t] <= P_BESS_dis_inst, name=f"B5_InstalledDischargePower[{t}]")
        m.addConstr(P_BESS_dis[t] <= bess_discharge_bound * (1 - x_BESS[t]), name=f"B5_DischargeMode[{t}]")
        m.addConstr(P_BESS_dis[t] <= bess_discharge_c_rate * Q_BESS, name=f"B5_DischargeCRate[{t}]")

    # B7. BESS State-of-Energy Balance
    for t_idx in range(1, len(times)):
        t, t_prev = times[t_idx], times[t_idx - 1]
        m.addConstr(E_BESS[t] == (1 - bess_self_discharge) * E_BESS[t_prev] + bess_charge_efficiency * P_BESS_ch[t] * dt - P_BESS_dis[t] * dt / bess_discharge_efficiency, name=f"B7_BESSEnergyBalance[{t}]")

    # B8. Initial and Terminal BESS Conditions
    m.addConstr(E_BESS[first_t] == bess_initial_soc * Q_BESS, name="B8_BESSInitialEnergy")
    m.addConstr(E_BESS[last_t] >= E_BESS[first_t], name="B8_BESSTerminalEnergy")

    # ============================================================
    # PV Constraints
    # ============================================================

    # PV1. Installed PV Capacity
    m.addConstr(E_PV_cap <= pv_max_kw, name="PV1_MaxPVCapacity")
    m.addConstr(E_PV_cap >= pv_min, name="PV_Installed")    #Change

    # PV2. PV Generation
    for t in times:
        m.addConstr(P_PV_generated[t] == E_PV_cap * pv_profiles_t[t], name=f"PV2_PVGeneration[{t}]")

    # ============================================================
    # 9. Grid Connection and Import / Export Constraints
    # ============================================================

    # G1. Grid Connection Capacity
    m.addConstr(S_G >= existing_grid_capacity, name="G1_ExistingGridCapacity")
    m.addConstr(S_G <= max_grid_capacity, name="G1_MaxGridCapacity")

    # G2. Grid Import Limited by Selected Connection
    for t in times:
        m.addConstr(P_G_imp[t] <= S_G, name=f"G2_GridImportConnection[{t}]")

    # G3. Grid Export Limited by Selected Connection
    for t in times:
        m.addConstr(P_G_exp[t] <= S_G, name=f"G3_GridExportConnection[{t}]")

    # G4. Grid Import-Mode and Network Limit
    for t in times:
        m.addConstr(P_G_imp[t] <= grid_import_bound * x_G[t], name=f"G4_GridImportMode[{t}]")

    # G5. Grid Export-Mode and Network Limit
    for t in times:
        m.addConstr(P_G_exp[t] <= grid_export_bound * (1 - x_G[t]), name=f"G5_GridExportMode[{t}]")

    # ============================================================
    # 10. Depot Electrical Power Balance (matrix)
    # ============================================================

    # E1. Power Conservation at the Depot Bus
    P_G_imp_vec    = gp.MVar.fromlist([P_G_imp[t] for t in times])
    P_PV_vec       = gp.MVar.fromlist([P_PV_generated[t] for t in times])
    P_BESS_dis_vec = gp.MVar.fromlist([P_BESS_dis[t] for t in times])
    P_BESS_ch_vec  = gp.MVar.fromlist([P_BESS_ch[t] for t in times])
    P_G_exp_vec    = gp.MVar.fromlist([P_G_exp[t] for t in times])
    aux_vec        = np.array([aux_load_t[t] for t in times])

    m.addConstr(
        P_G_imp_vec + P_PV_vec + P_BESS_dis_vec
        == P_SB_ch.sum(axis=1) + P_BESS_ch_vec + P_G_exp_vec + aux_vec,
        name="E1_DepotPowerBalance"
    )


    # =====================================================
    # 11. Network Charges
    # =====================================================
    
    is_peak = df_data['peak'].to_numpy()
    is_shoulder = df_data['shoulder'].to_numpy()
    net_energy_rate = df_data['network_energy_rate'].to_numpy()

    demand_peak = m.addVars(months, lb=dict_limits['min_demand_charge'], name="demand_peak")
    demand_shoulder = m.addVars(months, lb=dict_limits['min_demand_charge'], name="demand_shoulder")
    demand_offpeak = m.addVars(months, lb=dict_limits['min_demand_charge'], name="demand_offpeak")

    for month in range(months):
        t_start = month_limits[month - 1] + 1 if month > 0 else 0
        t_end = month_limits[month] + 1

        for t_pos in range(t_start, t_end):
            t_key = times[t_pos]
            if is_peak[t_pos]:
                m.addConstr(demand_peak[month] >= P_G_imp[t_key], name=f"NetDemandPeak[{month},{t_pos}]")
            elif is_shoulder[t_pos]:
                m.addConstr(demand_shoulder[month] >= P_G_imp[t_key], name=f"NetDemandShoulder[{month},{t_pos}]")
            else:
                m.addConstr(demand_offpeak[month] >= P_G_imp[t_key], name=f"NetDemandOffpeak[{month},{t_pos}]")

    network_energy_cost = gp.quicksum(
        P_G_imp[times[t_pos]] * dt * net_energy_rate[t_pos]
        for t_pos in range(len(times))
    )

    an_op_network_demand = gp.quicksum(
        dict_limits['demand_rate_peak'] * demand_peak[month]
        + dict_limits['demand_rate_shoulder'] * demand_shoulder[month]
        + dict_limits['demand_rate_offpeak'] * demand_offpeak[month]
        for month in range(months)
    )

    # ============================================================
    # Objective Function Definition
    # ============================================================

    # OBJECTIVE 1: COSTS
    # ============================================================
    # OPERATION
    # ============================================================

    # Annual cost of electricity imported from grid
    if whole_market == 1:
        an_op_grid_imp = gp.quicksum(P_G_imp[t] * dt * spot_price_t[t] / 1000 for t in times)
    else:
        an_op_grid_imp = gp.quicksum(P_G_imp[t] * dt * grid_price_t[t] for t in times)

    # Annual revenue from electricity exported to grid
    if whole_market == 1:
        an_op_grid_exp = gp.quicksum(P_G_exp[t] * dt * spot_price_t[t] / 1000 for t in times)
    else:
        an_op_grid_exp = gp.quicksum(P_G_exp[t] * dt * feed_in_tariff_t[t] for t in times)

    # Diesel truck operating cost
    # route_diesel_used[r] = litres consumed per route
    # diesel_price_t[y] = AUD/L
    an_op_diesel = gp.quicksum(
        route_diesel_used[r]* diesel_price_t[route_year[r]]* (1 - z[route_truck[r], route_year[r]]) for r in routes)

    # Total annual operating cost
    an_op_network = an_op_network_demand + network_energy_cost + network_access_cost

    an_op = (an_op_grid_imp - an_op_grid_exp + an_op_diesel + an_op_network)


    # ============================================================
    # CAPITAL AND INVESTMENTS
    # ============================================================

    # Capital recovery factor
    crf = (discount_rate * (1 + discount_rate) ** project_lifetime / ((1 + discount_rate) ** project_lifetime - 1))

    # Swappable batteries
    an_capex_SB = (crf * battery_cost * N_SB)

    # Chargers
    an_capex_CS = (crf * charger_cost * N_CS)

    # Stationary BESS
    an_capex_BESS = (crf * bess_capex_per_kwh * Q_BESS)

    # PV
    an_capex_PV = (crf * pv_capex_per_kw * E_PV_cap)

    # Truck retrofits
    an_capex_retrofit = crf * retrofit_cost * (gp.quicksum(k[i, y] for i in trucks for y in years)+ sum(truck_base_electric.values())
)

    # Total annualised CAPEX
    an_capex = (an_capex_SB + an_capex_CS + an_capex_BESS + an_capex_PV + an_capex_retrofit)


    # ============================================================
    # REPLACEMENTS
    # ============================================================

    # Swappable batteries
    an_rep_SB = (replacement_annualisation(battery_cost, project_lifetime, discount_rate, sb_lifetime) * N_SB)

    # Chargers
    an_rep_CS = (replacement_annualisation(charger_cost, project_lifetime, discount_rate, charger_lifetime) * N_CS)

    # Stationary BESS
    an_rep_BESS = (replacement_annualisation(bess_capex_per_kwh, project_lifetime, discount_rate, bess_lifetime) * Q_BESS)

    # PV
    an_rep_PV = (replacement_annualisation(pv_capex_per_kw, project_lifetime, discount_rate, pv_lifetime) * E_PV_cap)

    an_rep_retrofit_per_truck = replacement_annualisation(retrofit_cost, project_lifetime, discount_rate, retrofit_lifetime)
    an_rep_diesel_per_truck = replacement_annualisation(diesel_capex, project_lifetime, discount_rate, diesel_lifetime)
    an_rep_retrofit = gp.quicksum(an_rep_retrofit_per_truck * N_EV[y] for y in years)
    an_rep_diesel = gp.quicksum(an_rep_diesel_per_truck * N_D[y] for y in years)


    # Total annualised replacement costs
    an_rep = (an_rep_SB + an_rep_CS + an_rep_BESS + an_rep_PV + an_rep_retrofit + an_rep_diesel)


    # ============================================================
    # FIXED O&M
    # ============================================================

    # Swappable batteries
    an_om_SB = (battery_cost * N_SB * sb_om)

    # Chargers
    an_om_CS = (charger_cost * N_CS * charger_om)

    # Stationary BESS
    an_om_BESS = (bess_capex_per_kwh * Q_BESS * bess_om)

    # PV
    an_om_PV = (pv_capex_per_kw * E_PV_cap * pv_om)

    # Electric truck retrofit O&M
    an_om_retrofit = (retrofit_cost * retrofit_om * gp.quicksum(z[i, y] for i in trucks for y in years))

    # Diesel O&M
    an_om_DIESEL = gp.quicksum(diesel_capex * diesel_om * (1 - z[i, y]) for y in years for i in trucks_in_year[y])

    # Total annual O&M cost
    an_om = (an_om_SB + an_om_CS + an_om_BESS + an_om_PV + an_om_retrofit + an_om_DIESEL)


    # ============================================================
    # TOTAL COST OBJECTIVE
    # ============================================================

    obj1 = (an_op + an_capex + an_rep + an_om)
    m.setObjective(obj1, GRB.MINIMIZE)


    # ============================================================
    # Warm Start - diesel baseline
    # ============================================================
    # A cheap known-feasible point: no trucks converted, no batteries,
    # chargers, PV or BESS purchased. Note (from the last log): this
    # produced the exact same objective the solver's own heuristic
    # found unaided, so it's not currently doing useful work - left in
    # as harmless scaffolding for a more informative warm start later.
    for i in trucks:
        for y in years:
            z[i, y].Start = 0
            k[i, y].Start = 0
    w.Start = np.zeros(NB)
    a.Start = np.zeros((NR, NB))
    x_SB_ch.Start = np.zeros((NT, NB))
    E_SB.Start = np.zeros((NT, NB))
    P_SB_ch.Start = np.zeros((NT, NB))
    N_SB.Start = 0
    N_CS.Start = 0
    Q_BESS.Start = 0
    P_BESS_ch_inst.Start = 0
    P_BESS_dis_inst.Start = 0
    E_PV_cap.Start = 0


    # ============================================================
    # Optimise
    # ============================================================

    start_time = time.time()

    m.optimize()

    end_time = time.time()
    solve_time_seconds = end_time - start_time
    print(f"Optimisation: {solve_time_seconds:.2f} seconds")

    # Diagnostics - compare Runtime/NodeCount/MIPGap against prior runs.
    print(f"Runtime: {m.Runtime:.2f}s")
    print(f"NodeCount: {m.NodeCount}")
    print(f"NumVars: {m.NumVars}, NumConstrs: {m.NumConstrs}, NumBinVars: {m.NumBinVars}")
    print(f"MIPGap: {m.MIPGap:.4f}")



    # ============================================================
    # Store Optimal Solution
    # ============================================================

    # Accept both a proven-optimal solve and a time-limited solve that
    # still found a feasible solution.
    if m.status == GRB.OPTIMAL or (m.status == GRB.TIME_LIMIT and m.SolCount > 0):

        results_year = {

            # Objective
            "total_cost": obj1.getValue(),

            # Technology selection
            "number_of_batteries": N_SB.X,
            "number_of_chargers": N_CS.X,
            "number_of_retrofits": round(N_EV[years[-1]].X),
            "truck_base_electric": {i: int(round(z[i, years[-1]].X)) for i in trucks},
            "BESS_capacity_kWh": Q_BESS.X,
            "BESS_charge_capacity_kW": P_BESS_ch_inst.X,
            "BESS_discharge_capacity_kW": P_BESS_dis_inst.X,
            "PV_capacity_kW": E_PV_cap.X,
            "grid_capacity_kW": S_G.X,

            # Energy totals
            "grid_import_kWh": sum(P_G_imp[t].X * dt for t in times),
            "grid_export_kWh": sum(P_G_exp[t].X * dt for t in times),
            "PV_generation_kWh": sum(P_PV_generated[t].X * dt for t in times),
            "BESS_charge_kWh": sum(P_BESS_ch[t].X * dt for t in times),
            "BESS_discharge_kWh": sum(P_BESS_dis[t].X * dt for t in times),
            # P_SB_ch is an MVar (NT,NB) - sum the full solution array
            # rather than looping dict-style keys.
            "SB_charge_kWh": float(P_SB_ch.X.sum() * dt),

            # Cost breakdown
            "operation_cost": an_op.getValue(),
            "annualised_capex": an_capex.getValue(),
            "replacement_cost": an_rep.getValue(),
            "OM_cost": an_om.getValue(),

            # Solver
            "MIP_gap_percent": m.MIPGap * 100,
            "status": "Optimal" if m.status == GRB.OPTIMAL else "Time limit reached (feasible)",
            "solve_time_seconds": solve_time_seconds,
        }

        # Convert to dataframe
        df_results_year = pd.DataFrame([results_year])

        # Print
        print("\n========================================")
        print("OPTIMAL SOLUTION YEAR 2" if m.status == GRB.OPTIMAL else "FEASIBLE SOLUTION (TIME LIMIT REACHED)")
        print("========================================")

        for key, value in results_year.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value}")

        print("========================================\n")

        return results_year, df_results_year

    else:
        # No feasible solution found (e.g. infeasible model, or time
        # limit hit with zero incumbent solutions)
        print(f"\nNo feasible solution found. Solver status: {m.status}\n")
        return None, None

def replacement_annualisation(
    unit_cost,
    project_lifetime,
    discount_rate,
    technology_lifetime
):

    replacement_present_value = 0

    replacement_year = technology_lifetime

    while replacement_year < project_lifetime:

        replacement_present_value += (
            unit_cost
            / (1 + discount_rate) ** replacement_year
        )

        replacement_year += technology_lifetime

    crf = (
        discount_rate * (1 + discount_rate) ** project_lifetime
        / ((1 + discount_rate) ** project_lifetime - 1)
    )

    return replacement_present_value * crf
