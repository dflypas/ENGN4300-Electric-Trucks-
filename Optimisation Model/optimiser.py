#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:57:48 2026

"""

"""
DairyDrive Electric - Swappable Battery Optimisation Module


Core Assumptions
--------------------------
1. Every route must be assigned exactly one swappable battery.
2. A battery cannot be assigned to more than one active route at the same time.
3. Battery discharges at constant rate while on a route. 
4. Charger charges at constant rate while actively charging.
5. Route energy is considered taken at the departure timestep.
6. Battery state of charge is constrained between the specified minimum and
   maximum SOC limits, and is initially defined. 
7. Charging is only possible when a battery is at the depot and a charger is available. 
8. Route Energy requirements are supplied externally through the route input
   data.
9. The optimiser selects both the number of batteries and the number of
   chargers required to maintain a feasible operating schedule.
    


Decision variables
------------------
sb_active[b]
    Binary variable equal to 1 if swappable battery b is needed in system/ever used on a route. 

N_CS
    Integer variable representing the number of chargers installed.

assign_SB[r,b]
    Binary variable equal to 1 if battery b is assigned to route r.

x_SB_charge[b,t]
    Binary variable equal to 1 if battery b is charging at timestep t.

f_SB[b,t]
    Continuous variable representing charging energy supplied to battery b
    during timestep t, in kWh.

E_SB[b,t]
    Continuous variable representing the stored energy of battery b at
    timestep t, in kWh.



Main constraints
----------------
Route assignment: Each route must be completed by exactly one battery.

Battery activation: Only batteries that have been uniquely created may be used. 

Route overlap: A battery cannot complete two overlapping routes.

Charging availability: A battery cannot charge while it is away completing a route.

SOC limits: Battery energy must remain within the specified minimum and maximum state of charge.

Battery energy balance: Stored battery energy evolves according to previous SOC, charging energy, charging efficiency and route energy consumption.

Pre-departure feasibility: A battery must contain sufficient energy before departure to complete its assigned route while remaining above the minimum SOC requirement.

Charger capacity: The number of simultaneously charging batteries and total charging energy are limited by the number and power of installed chargers.




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

from route_data_processor import process_route_data


# ============================================================
# Inputs and Input Variables
# ============================================================

INPUT_CSV = "input_data.csv"
ROUTE_CSV = "route_schedule.csv"
PV_CSV = "pv_profile_output.csv"


# ============================================================
# Read Model Input Data
# ============================================================

inputs_df = pd.read_csv(INPUT_CSV)

# Convert:
#
# field_name | value
# ------------------
# sb_capacity | 920
#
# into:
#
# inputs["sb_capacity"] = 920

inputs = (
    inputs_df
    .set_index("field_name")["value"]
    .to_dict()
)


# ============================================================
# Study Inputs
# ============================================================

duration_days = int(float(inputs["duration"]))

dt = float(inputs["time_step_dt"])


# ============================================================
# Swappable Battery Inputs
# ============================================================

sb_cap = float(inputs["sb_max_capacity"])

sb_min_soc = float(inputs["sb_min_soc"])

sb_max_soc = float(inputs["sb_max_soc"])

sb_initial_soc = float(inputs["sb_initial_soc"])

charge_efficiency = float(
    inputs["sb_charge_efficiency"]
)

discharge_efficiency = float(
    inputs["sb_discharge_efficiency"]
)

BATTERY_COST = float(
    inputs["battery_cost"]
)

# ============================================================
# Charger Inputs
# ============================================================

charger_power_kw = float(
    inputs["charger_power"]
)

CHARGER_COST = float(
    inputs["charger_cost"]
)


# ============================================================
# Stationary Battery Inputs
# ============================================================

soc_BESS_initial = float(
    inputs["BESS_initial_soc"]
)

BESS_min_soc = float(
    inputs["BESS_min_soc"]
)

BESS_max_soc = float(
    inputs["BESS_max_soc"]
)


# ============================================================
# Derived Parameters
# ============================================================

E_SB_min = sb_min_soc * sb_cap

E_SB_max = sb_max_soc * sb_cap

E_SB_initial = sb_initial_soc * sb_cap


# Maximum energy that one charger can supply during one timestep
max_sb_charge_per_step = charger_power_kw * dt


def optimiser():
    # ============================================================
    # Import Route and Time Data
    # ============================================================
    (routes_df, routes, dates, times, first_t, last_t, prev_t, 
     route_departure, routes_ending, route_energy, route_active, 
     routes_departing_at, batteries, chargers) = process_route_data(ROUTE_CSV)
    
    
    pv_generated = "pv_profile_output.csv"
    pv_generated_df = pd.read_csv(pv_generated)
    pv_values = list(pv_generated_df["PV_Profile"]*1000)
    pv_generated_t = dict(zip(times, pv_values))
    
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
    f_SB = m.addVars(batteries, times, vtype=GRB.CONTINUOUS, lb=0, ub=charger_power_kw, name="charge_energy_amount")
    E_SB = m.addVars(batteries, times, vtype=GRB.CONTINUOUS, lb=0, ub=E_SB_max, name="battery_energy_level")
    
    
    
    
    
    # Stationary (Depot) Battery Energy, Charging and Discharging Amount
    E_BESS_cap = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Capacity_BESS_kWh")
    E_BESS = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="Energy_BESS_kWh")
    
    # PV Energy Sold and Grid Import Variables
    grid_import = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="Grid_import")
    pv_export = m.addVars(times, vtype=GRB.CONTINUOUS, lb=0, name="PV_export")
    
    
    
    
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
            away_expr = gp.quicksum(assign_SB[r, b] * int(route_active.get((r, t), 0)) for r in routes)
            m.addConstr(away_expr <= sb_active[b], name=f"SB_NoRouteOverlap[{b},{t}]")
            m.addConstr(x_SB_charge[b, t] + away_expr <= sb_active[b], name=f"SB_NoChargeWhileAway[{b},{t}]",)
    
    
    # ============================================================
    # 2. Swappable-battery SOC, route-energy, and charging constraints (DECLAN)
    # ============================================================
    
    for b in batteries:
        for t in times:
            # Storage bounds switch off automatically when the candidate battery is not purchased.
            m.addConstr(E_SB[b, t] >= 0.2*sb_cap * sb_active[b], name=f"SB_SOCMin[{b},{t}]")
            m.addConstr(E_SB[b, t] <= sb_cap * sb_active[b], name=f"SB_SOCMax[{b},{t}]")
    
            # Charging is possible only for an active battery and one charger slot.
            m.addConstr(x_SB_charge[b, t] <= sb_active[b], name=f"SB_ChargeOnlyIfActive[{b},{t}]")
            m.addConstr(f_SB[b, t] <= max_sb_charge_per_step * x_SB_charge[b, t], name=f"SB_ChargeRate[{b},{t}]",)
    
            # Route-energy output occurs at the route departure event.
            route_energy_used = gp.quicksum(route_energy[r] * assign_SB[r, b] for r in routes if routes_departing_at[r] == t)
    
            previous_energy = E_SB_initial * sb_active[b] if t == first_t else E_SB[b, prev_t[t]]
    
            # SOC state equation. If route_energy_kwh is already battery-side
            # consumption, set sb_discharge_efficiency = 1.0 in the input data.
            m.addConstr(E_SB[b, t] == previous_energy 
                        + charge_efficiency * f_SB[b, t] 
                        - route_energy_used / discharge_efficiency, 
                        name=f"SB_EnergyBalance[{b},{t}]")
    
    # Energy Required to be assigned on a route
    for r in routes:
        dep = route_departure[r]
        required_battery_draw = route_energy[r] / discharge_efficiency
        for b in batteries:
            energy_before_departure = (E_SB_initial * sb_active[b] if dep == first_t else E_SB[b, prev_t[dep]])
            m.addConstr(energy_before_departure >= E_SB_min * sb_active[b] + required_battery_draw * assign_SB[r, b],
                name=f"SB_EnoughEnergyBeforeRoute[{r},{b}]",)
    
    # Each battery charging uses 1 charger
    for t in times:
        m.addConstr(gp.quicksum(x_SB_charge[b, t] for b in batteries) <= N_CS,name=f"Charger_Count[{t}]",)
        m.addConstr(gp.quicksum(f_SB[b, t] for b in batteries) <= N_CS * charger_power_kw * dt,name=f"Charger_TotalEnergy[{t}]",)
    
    
    # ============================================================
    # 3. Stationary Battery Constraints (ZARIF & DECLAN)
    # ============================================================
    
    
    
    for t in times:
        
        m.addConstr(E_BESS[t] >= 0.2 * E_BESS_cap, name=f"BESS_SOCMin[{t}]")
        m.addConstr(E_BESS[t] <= E_BESS_cap, name=f"BESS_SOCMax[{t}]")
        
        previous_BESS_energy = soc_BESS_initial * E_BESS_cap if t == first_t else E_BESS[prev_t[t]]
    
        SB_charging_energy = gp.quicksum(f_SB[b, t] for b in batteries)
    
        m.addConstr(E_BESS[t] == previous_BESS_energy + grid_import[t] + pv_generated_t[t] - pv_export[t] - SB_charging_energy ,name=f"BESS_EnergyBalance[{t}]")
    
    
    # ============================================================
    # Objective Function Definition
    # ============================================================
    
    m.setObjective(
        BATTERY_COST * gp.quicksum(sb_active[b] for b in batteries)
        + CHARGER_COST * N_CS,
        GRB.MINIMIZE,
    )
    
    m.optimize()
    
    
    


    # ============================================================
    # Visualise Results
    # ============================================================
    
    import matplotlib.pyplot as plt
    
    print(E_BESS_cap.X)
    
    if m.status == GRB.OPTIMAL:
    
        n_batteries = sum(sb_active[b].X for b in batteries)
        n_chargers = N_CS.X
        print(f"Batteries needed: {n_batteries:.0f}")
        print(f"Chargers needed:  {n_chargers:.0f}")
    
    
        active_batteries = [b for b in batteries if sb_active[b].X > 0.5]
    
        plt.figure(figsize=(14, 7))
        
        for b in active_batteries:
            battery_times = []
            battery_soc = []
        
            for t in times:
                battery_times.append(t)
                battery_soc.append((E_SB[b, t].X / sb_cap) * 100)
        
            plt.plot(battery_times, battery_soc, label=f"Battery {b}")
        
        plt.xlabel("Time")
        plt.ylabel("Battery State of Charge (%)")
        plt.title("Swappable Battery State of Charge Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return
