#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:21:23 2026

@author: mollyrobertson
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time



def optimiser(
    routes, trucks, times, route_truck, active_routes_at, route_energy, route_departure, route_return, overlapping_routes,
    pv_profiles_t, spot_price_t, sell_price_t, grid_price_t,
    
    diesel_truck_consumption, diesel_cost_per_kwh, diesel_price_t,
    
    duration_days, dt, years, first_t, last_t, batteries, chargers,
    n_trucks, truck_consumption,
    
    sb_cap, sb_min_soc, sb_max_soc, sb_initial_soc, sb_charge_efficiency,
    sb_discharge_efficiency, sb_self_discharge, sb_max_charge_power,
    
    battery_cost, charger_power_kw, charger_cost,
    
    bess_initial_soc, bess_min_soc, bess_max_soc, bess_max_cap,
    bess_max_charge_power, bess_max_discharge_power,
    bess_charge_efficiency, bess_discharge_efficiency, bess_self_discharge,
    bess_charge_c_rate, bess_discharge_c_rate,
    
    max_grid_import, max_grid_export,
    existing_grid_capacity, max_grid_capacity,
    
    pv_efficiency, pv_max_kw, pv_capex_per_kw,
    
    bess_capex_per_kwh, retrofit_cost, diesel_capex,
    
    discount_rate, project_lifetime,
    
    sb_lifetime, charger_lifetime, bess_lifetime, pv_lifetime, retrofit_lifetime,
    
    sb_om, charger_om, bess_om, pv_om, retrofit_om, diesel_om,
    
    route_year,             # y(r)
    trucks_in_year,         # I(y)
    fleet_size,             # N_fleet(y)
    truck_first_year,       # y_first(i)
    truck_base_electric,    # z_base(i)
    
    routes_departing_at,    # D(t)
    route_pre_departure,    # t_pre(r)
    
    aux_load_t,             # P_aux(t)
):
    
    
    
    # ============================================================
    # Model
    # ============================================================
    
    m = gp.Model("battery_selection")
    m.setParam('Presolve', 2)
    m.setParam("MIPGap", 0.05)
    
    
    
    # ============================================================
    # Decision Variables
    # ============================================================
    
    N_EV = m.addVars(years, vtype=GRB.INTEGER, lb=0, name="N_EV")
    N_D = m.addVars(years, vtype=GRB.INTEGER, lb=0, name="N_D")
    N_SB = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(batteries), name="N_SB")
    N_CS = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(chargers), name="N_CS")
    
    z = m.addVars(trucks, years, vtype=GRB.BINARY, name="z")
    k = m.addVars(trucks, years, vtype=GRB.BINARY, name="k")
    w = m.addVars(batteries, vtype=GRB.BINARY, name="w")
    a = m.addVars(routes, batteries, vtype=GRB.BINARY, name="a")
    x_SB_ch = m.addVars(batteries, times, vtype=GRB.BINARY, name="x_SB_ch")
    
    E_SB = m.addVars(batteries, times, vtype=GRB.CONTINUOUS, lb=0, name="E_SB")
    P_SB_ch = m.addVars(batteries, times, vtype=GRB.CONTINUOUS, lb=0, name="P_SB_ch")
    
    Q_BESS = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Q_BESS")
    P_BESS_ch_inst = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P_BESS_ch_inst")
    P_BESS_dis_inst = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="P_BESS_dis_inst")
    E_BESS = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="E_BESS")
    P_BESS_ch = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_BESS_ch")
    P_BESS_dis = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_BESS_dis")
    x_BESS = m.addVars(times, vtype=GRB.BINARY, name="x_BESS")

    E_PV_cap = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=pv_max_kw, name="E_PV_cap")
    P_PV_generated = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_PV_generated")    
    
    S_G = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S_G")
    P_G_imp = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_G_imp")
    P_G_exp = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="P_G_exp")
    x_G = m.addVars(times, vtype=GRB.BINARY, name="x_G")
    
    # ============================================================
    # 3. Fleet Conversion Constraints
    # ============================================================
    
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
    # 5. Swappable Battery Inventory and Route Assignment
    # ============================================================
    
    # SB1. Number of Swappable Batteries Purchased
    m.addConstr(N_SB == gp.quicksum(w[b] for b in batteries), name="SB1_NumberSwappableBatteries")
    
    # SB2. One Battery per Electric Route
    for r in routes: 
        m.addConstr(gp.quicksum(a[r, b] for b in batteries) == z[route_truck[r], route_year[r]], name=f"SB2_OneBatteryPerElectricRoute[{r}]")
    
    # SB3. Only Purchased Batteries May Be Assigned
    for r in routes:
        for b in batteries: 
            m.addConstr(a[r, b] <= w[b], name=f"SB3_OnlyPurchasedBattery[{r},{b}]")
    
    # SB4. A Battery Cannot Serve Overlapping Routes
    for b in batteries:
        for t in times: 
            m.addConstr(gp.quicksum(a[r, b] for r in active_routes_at[t]) <= w[b], name=f"SB4_NoOverlappingRoutes[{b},{t}]")
         
    # ============================================================
    # 6. Swappable Battery Charging and Charger Assignment
    # ============================================================
    
    # Battery can only charge if purchased
    for b in batteries:
        for t in times:
            m.addConstr(x_SB_ch[b, t] <= w[b], name=f"C1_OnlyPurchasedBattery[{b},{t}]")
    
    # Battery cannot charge while on a route
    for b in batteries:
        for t in times:
            m.addConstr(x_SB_ch[b, t] + gp.quicksum(a[r, b] for r in active_routes_at[t]) <= w[b], name=f"C2_NoChargeWhileRoute[{b},{t}]")
    
    # Battery charging power limit
    for b in batteries:
        for t in times:
            m.addConstr(P_SB_ch[b, t] <= sb_max_charge_power * x_SB_ch[b, t], name=f"C3_BatteryChargePower[{b},{t}]")
    
    # Number of batteries charging cannot exceed number of chargers
    for t in times:
        m.addConstr(gp.quicksum(x_SB_ch[b, t] for b in batteries) <= N_CS, name=f"C4_ChargerCount[{t}]")
    
    # Total charging power cannot exceed installed charger power
    for t in times:
        m.addConstr(gp.quicksum(P_SB_ch[b, t] for b in batteries) <= N_CS * charger_power_kw, name=f"C5_TotalChargerPower[{t}]")
    # ============================================================
    # 7. Swappable Battery State of Energy
    # ============================================================
    
    # SOC1. Stored-Energy Bounds
    for b in batteries:
        for t in times:
            m.addConstr(E_SB[b, t] >= sb_min_soc * sb_cap * w[b], name=f"SOC1_SB_Min[{b},{t}]")
            m.addConstr(E_SB[b, t] <= sb_max_soc * sb_cap * w[b], name=f"SOC1_SB_Max[{b},{t}]")
    
    # SOC2. Swappable-Battery State-of-Energy Balance
    for b in batteries:
        for t_idx in range(1, len(times)):
            t, t_prev = times[t_idx], times[t_idx - 1]
            m.addConstr(E_SB[b, t] == (1 - sb_self_discharge) * E_SB[b, t_prev] + sb_charge_efficiency * P_SB_ch[b, t] * dt - gp.quicksum(route_energy[r] * a[r, b] for r in routes_departing_at[t]), name=f"SOC2_SB_EnergyBalance[{b},{t}]")
    
    # SOC3. Sufficient Energy Before Departure
    for r in routes:
        for b in batteries: m.addConstr(E_SB[b, route_pre_departure[r]] >= sb_min_soc * sb_cap * w[b] + route_energy[r] * a[r, b], name=f"SOC3_SufficientDepartureEnergy[{r},{b}]")
    
    # SOC4. Initial Swappable-Battery Energy
    for b in batteries: 
        m.addConstr(E_SB[b, first_t] == sb_initial_soc * sb_cap * w[b], name=f"SOC4_InitialSBEnergy[{b}]")
    
    # SOC5. Terminal Swappable-Battery Condition
    for b in batteries: 
        m.addConstr(E_SB[b, last_t] >= E_SB[b, first_t], name=f"SOC5_TerminalSBEnergy[{b}]")
         
    # ============================================================
    # 8. Stationary BESS Constraints
    # ============================================================
    
    # B1. Installed BESS Energy Capacity
    m.addConstr(Q_BESS <= bess_max_cap, name="B1_BESSCapacity")
    
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
        m.addConstr(P_BESS_ch[t] <= bess_max_charge_power * x_BESS[t], name=f"B4_ChargeMode[{t}]")
        m.addConstr(P_BESS_ch[t] <= bess_charge_c_rate * Q_BESS, name=f"B4_ChargeCRate[{t}]")
    
    # B5. BESS Discharging-Power Limits
    for t in times:
        m.addConstr(P_BESS_dis[t] <= P_BESS_dis_inst, name=f"B5_InstalledDischargePower[{t}]")
        m.addConstr(P_BESS_dis[t] <= bess_max_discharge_power * (1 - x_BESS[t]), name=f"B5_DischargeMode[{t}]")
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
        m.addConstr(P_G_imp[t] <= max_grid_import * x_G[t], name=f"G4_GridImportMode[{t}]")
    
    # G5. Grid Export-Mode and Network Limit
    for t in times: 
        m.addConstr(P_G_exp[t] <= max_grid_export * (1 - x_G[t]), name=f"G5_GridExportMode[{t}]")


    # ============================================================
    # 10. Depot Electrical Power Balance
    # ============================================================
    
    # E1. Power Conservation at the Depot Bus
    for t in times: 
        m.addConstr(P_G_imp[t] + P_PV_generated[t] + P_BESS_dis[t] == gp.quicksum(P_SB_ch[b, t] for b in batteries) + P_BESS_ch[t] + P_G_exp[t] + aux_load_t[t], name=f"E1_DepotPowerBalance[{t}]")
    
    
    
    
    # ============================================================
    # Objective Function Definition
    # ============================================================
    start_time = time.time()
    
    # OBJECTIVE 1: COSTS
    # ============================================================
    # OPERATION
    # ============================================================

    # Annual cost of electricity imported from grid
    an_op_grid_imp = gp.quicksum(P_G_imp[t] * dt * grid_price_t[t] for t in times)
    
    # Annual revenue from electricity exported to grid
    an_op_grid_exp = gp.quicksum(P_G_exp[t] * dt * spot_price_t[t] / 1000 for t in times)
    
    # Diesel truck energy cost
    an_op_diesel = gp.quicksum((route_return[r] - route_departure[r]) * diesel_truck_consumption * diesel_price_t[route_year[r]] * (1 - z[route_truck[r], route_year[r]]) for r in routes)
    
    # Total annual operating cost
    an_op = (an_op_grid_imp - an_op_grid_exp + an_op_diesel)
    
    
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
    an_capex_retrofit = (crf * retrofit_cost * gp.quicksum(k[i, y] for i in trucks for y in years))
    
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
    
    # Truck retrofits
    an_rep_retrofit = (replacement_annualisation(retrofit_cost, project_lifetime, discount_rate, retrofit_lifetime) * gp.quicksum(k[i, y] for i in trucks for y in years))
    
    # Total annualised replacement costs
    an_rep = (an_rep_SB + an_rep_CS + an_rep_BESS + an_rep_PV + an_rep_retrofit)
    
    
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
    # Optimise
    # ============================================================
    
    start_time = time.time()
    
    m.optimize()
    
    end_time = time.time()
    print(f"Optimisation: {end_time - start_time:.2f} seconds")
        
        
        

    # ============================================================
    # Store Optimal Solution
    # ============================================================
    
    if m.status == GRB.OPTIMAL:
    
        results = {
    
            # Objective
            "total_cost": obj1.getValue(),
    
            # Technology selection
            "number_of_batteries": N_SB.X,
            "number_of_chargers": N_CS.X,
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
            "SB_charge_kWh": sum(
                P_SB_ch[b, t].X * dt
                for b in batteries
                for t in times
            ),
    
            # Cost breakdown
            "operation_cost": an_op.getValue(),
            "annualised_capex": an_capex.getValue(),
            "replacement_cost": an_rep.getValue(),
            "OM_cost": an_om.getValue(),
    
            # Solver
            "MIP_gap_percent": m.MIPGap * 100
        }
    
        # Convert to dataframe
        df_results = pd.DataFrame([results])
    
        # Print
        print("\n========================================")
        print("OPTIMAL SOLUTION")
        print("========================================")
    
        for key, value in results.items():
            print(f"{key}: {value:,.2f}")
    
        print("========================================\n")
    
        # Export
        df_results.to_csv(
            "optimal_results.csv",
            index=False
        )
    
        return df_results

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







        