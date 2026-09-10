#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:57:48 2026

"""

"""
DairyDrive Electric - Swappable Battery Optimisation Module


CURRENT Objective
---------
Minimise total battery and charger cost: 
    battery cost × number of selected batteries + charger cost × number of selected chargers



Outputs
-------
    - Minimum number of swappable batteries required
    - Minimum number of chargers required
    - Battery state of charge over time
    - Battery-to-route assignments

"""


import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def optimiser(routes, times, first_t, prev_t, batteries, active_routes_at, truck_consumption,
    dt, sb_cap, E_SB_min, E_SB_max, E_SB_initial, charge_efficiency, discharge_efficiency, 
    battery_cost, charger_power_kw, charger_cost, max_sb_charge_per_step, soc_BESS_initial,
    pv_per_panel_t, retrofit_cost, panel_price, peak_panel, no_trucks, num_panels, panel_size,
    roof, BESS_Capex, peak_grid_price, offpeak_grid_price, feed_in_tariff, SB_maint,
    Charger_maint, BESS_maint, PV_maint, retrofit_maint, diesel_price_per_km, diesel_fleet_annual_km,
    Project_lt):
    
    
    # ============================================================
    # Model
    # ============================================================
    
    m = gp.Model("battery_selection")
    
    # ============================================================
    # Decision Variables
    # ============================================================
    
    # Batteries and Chargers Variables
    sb_active = m.addVars(batteries, vtype=GRB.BINARY, name="battery_active")
    N_CS = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(batteries), name="number_of_chargers")
    
    # Battery Depot, Route & Charging, Discharging State Variables
    # assign_SB | sb_active
    # 0 | 1 : At depot, charging
    # 0 | 0 : At depot, Idle
    # 1 | 1/0: On route --> Discharging
    assign_SB = m.addVars(routes, batteries, vtype=GRB.BINARY, name="on_route")
    x_SB_charge = m.addVars(batteries, times, vtype=GRB.BINARY, name="charging")
    
    # Battery Energy (Charge Amount and Energy Level) (kWh) 
    f_SB = m.addVars(batteries, times, vtype=GRB.CONTINUOUS, lb=0, ub=max_sb_charge_per_step, name="charge_energy_amount")
    E_SB = m.addVars(batteries, times, vtype=GRB.CONTINUOUS, lb=0, ub=E_SB_max, name="battery_energy_level")
    
    
    
    
    
    # Stationary (Depot) Battery Energy, Charging and Discharging Amount
    E_BESS_cap = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Capacity_BESS_kWh")
    E_BESS = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="Energy_BESS_kWh")
    
    # PV Energy Sold and Grid Import Variables
    grid_import = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="Grid_import")
    pv_export = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="PV_export")
    num_panels = m.addVar(vtype=GRB.INTEGER, lb = 0, name="Num_Panels")



    # Number of trucks left as a retrofit
    no_retrofit = m.addVar(vtype=GRB.INTEGER, lb = 0, name="Number_Retrofits")
    no_diesel = m.addVar(vtype=GRB.INTEGER, lb = 0, name="Number_Diesel")
    
    
    # ============================================================
    # Constraints 
    # ============================================================
    
    # ============================================================
    # 1. Route assignment and battery availability constraints (DECLAN)
    # ============================================================
    
    # Every route needs exactly one battery
    for r in routes:
        m.addConstr(gp.quicksum(assign_SB[r, b] for b in batteries) == 1, name=f"Route_OneBattery[{r}]")
        for b in batteries:
            m.addConstr(assign_SB[r, b] <= sb_active[b], name=f"Route_ActiveBattery[{r},{b}]")
    
    for b in batteries:
        for t in times:
            away_expr = gp.quicksum(assign_SB[r, b] for r in active_routes_at[t])
            m.addConstr(away_expr <= sb_active[b], name=f"SB_NoRouteOverlap[{b},{t}]")
            m.addConstr(x_SB_charge[b, t] + away_expr <= sb_active[b], name=f"SB_NoChargeWhileAway[{b},{t}]",)
    
    
    # ============================================================
    # 2. Swappable-battery SOC, route-energy, and charging constraints (DECLAN)
    # ============================================================
    
    for b in batteries:
        for t in times:
            # Storage bounds switch off automatically when the candidate battery is not purchased.
            m.addConstr(E_SB[b, t] >= E_SB_min * sb_active[b], name=f"SB_SOCMin[{b},{t}]")
            m.addConstr(E_SB[b, t] <= E_SB_max * sb_active[b], name=f"SB_SOCMax[{b},{t}]")
    
            # Charging is possible only for an active battery and one charger slot.
            m.addConstr(f_SB[b, t] <= max_sb_charge_per_step * x_SB_charge[b, t], name=f"SB_ChargeRate[{b},{t}]",)

            # Constant route energy is consumed during every hour the battery is on a route.
            away_expr = gp.quicksum(assign_SB[r, b] for r in active_routes_at[t])
            route_energy_used = (truck_consumption * dt* away_expr)

            previous_energy = E_SB_initial * sb_active[b] if t == first_t else E_SB[b, prev_t[t]]
    
            # SOC state equation. If route_energy_kwh is already battery-side
            # consumption, set sb_discharge_efficiency = 1.0 in the input data.
            m.addConstr(E_SB[b, t] == previous_energy 
                        + charge_efficiency * f_SB[b, t] 
                        - route_energy_used / discharge_efficiency, 
                        name=f"SB_EnergyBalance[{b},{t}]")
    
    # Each battery charging uses 1 charger
    for t in times:
        m.addConstr(gp.quicksum(x_SB_charge[b, t] for b in batteries) <= N_CS,name=f"Charger_Count[{t}]",)
        m.addConstr(gp.quicksum(f_SB[b, t] for b in batteries) <= N_CS * charger_power_kw * dt,name=f"Charger_TotalEnergy[{t}]",)
    
    
    # ============================================================
    # 3. Stationary Battery Constraints (ZARIF & DECLAN)
    # ============================================================
    
    
    pv_generated = {}

    for t in times:
        pv_generated[t] = pv_per_panel_t[t] * num_panels
        
        m.addConstr(E_BESS[t] >= 0.2 * E_BESS_cap, name=f"BESS_SOCMin[{t}]")
        m.addConstr(E_BESS[t] <= E_BESS_cap, name=f"BESS_SOCMax[{t}]")
        
        previous_BESS_energy = soc_BESS_initial * E_BESS_cap if t == first_t else E_BESS[prev_t[t]]
    
        SB_charging_energy = gp.quicksum(f_SB[b, t] for b in batteries)
    
        m.addConstr(E_BESS[t] == previous_BESS_energy + grid_import[t] + pv_generated[t] - pv_export[t] - SB_charging_energy ,name=f"BESS_EnergyBalance[{t}]")


    # =================================================
    # 4. Economic constraints (JOEL)
    # =================================================

    m.addConstr(no_retrofit + no_diesel == no_trucks, name="Truck_Count")
    m.addConstr(num_panels*panel_size <= roof)

    grid_price = {}
    for t in times:
        hour_of_day = t % 24
        if 15 <= hour_of_day < 21:
            grid_price[t] = peak_grid_price
        else:
            grid_price[t] = offpeak_grid_price
        
     
    # ============================================================
    # Objective Function Definition
    # ============================================================
    
    # ==============================
    # 1. Economic Definition (JOEL)
    # ==============================

    #This is essentially only here because the timescale is like 30 days. Can be comfortably removed if the timescale is increased past 1 year. 
    #If this is gone then project_lt needs to also be removed because otherwise the model freaks out.
    hours_in_horizon = len(times) * dt
    hours_per_year = 8760
    annualisation_factor = hours_per_year / hours_in_horizon


    Capex = gp.quicksum(sb_active[b] for b in batteries) * battery_cost + N_CS * charger_cost + no_retrofit * retrofit_cost + num_panels * panel_price * peak_panel + BESS_Capex * E_BESS_cap

    Opex = (
        Project_lt*annualisation_factor*(
            gp.quicksum(grid_price[t] * grid_import[t] for t in times) 
            - gp.quicksum(pv_export[t]*feed_in_tariff for t in times)
            )
        + Project_lt*(
            SB_maint*gp.quicksum(sb_active[b] for b in batteries)*battery_cost + Charger_maint*N_CS*charger_cost + BESS_maint * BESS_Capex * E_BESS_cap 
             + PV_maint * num_panels * panel_price * peak_panel + retrofit_maint*retrofit_cost*no_retrofit
             )
        + Project_lt*((diesel_fleet_annual_km/no_trucks)*diesel_price_per_km)*no_diesel
        )
    m.setObjective(Capex + Opex, GRB.MINIMIZE)

    m.Params.DualReductions = 0
    m.optimize()
    
    
    


    # ============================================================
    # Visualise Results
    # ============================================================
    
    import matplotlib.pyplot as plt
    
    
    
    if m.status == GRB.OPTIMAL:
        print(E_BESS_cap.X)
        price = m.ObjVal
        n_batteries = sum(sb_active[b].X for b in batteries)
        n_chargers = N_CS.X
        n_panels = num_panels.X
        n_retrofit = no_retrofit.X
        print(f"Batteries needed: {n_batteries:.0f}")
        print(f"Chargers needed:  {n_chargers:.0f}")
        print(f"Panels needed: {n_panels:.0f}")
        print(f"Retrofits completed: {n_retrofit:.0f}")
        print(f"Total price: {price:.0f}")
    
    
        # active_batteries = [b for b in batteries if sb_active[b].X > 0.5]
    
        # plt.figure(figsize=(14, 7))
        
        # for b in active_batteries:
        #     battery_times = []
        #     battery_soc = []
        
        #     for t in times:
        #         battery_times.append(t)
        #         battery_soc.append((E_SB[b, t].X / sb_cap) * 100)
        
        #     plt.plot(battery_times, battery_soc, label=f"Battery {b}")
        
        # plt.xlabel("Time")
        # plt.ylabel("Battery State of Charge (%)")
        # plt.title("Swappable Battery State of Charge Over Time")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

    return
