#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:58:06 2026

@author: mollyrobertson
"""
    
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ============================================================
# Parameters
# ============================================================

ROUTE_CSV = "route_schedule.csv"
routes_df = pd.read_csv(ROUTE_CSV)

BATTERY_CAPACITY_KWH = 920
INITIAL_SOC = 1
MINIMUM_SOC = 0.20
MAXIMUM_SOC = 1

CHARGE_POWER_KW_PER_HOUR = 250.0
CHARGE_EFFICIENCY = 0.95

CHARGE_BUFFER = 0.20


# ============================================================
# Assume Static Variables
# ============================================================


# ============================================================
# Create Sets
# ============================================================

# List of routes
routes = routes_df["route_no"].tolist()
end_time = routes_df["return_time"].max()

batteries = (range(1, len(routes)+1))
times = list(range(0, int(end_time) + 1))

# ============================================================
# Create Dictionaries
# ============================================================

departure_time = (routes_df.set_index("route_no")["departure_time"].to_dict())
return_time = (routes_df.set_index("route_no")["return_time"].to_dict())

routes_df["required_energy_kwh"] = (routes_df["route_energy_kwh"] * (1.0 + CHARGE_BUFFER))
route_energy_kwh = (routes_df.set_index("route_no")["required_energy_kwh"].to_dict())

active_routes = {t: [r for r in routes if departure_time[r] <= t < return_time[r]] for t in times}

# ============================================================
# CREATE GUROBI MODEL
# ============================================================

model = gp.Model("battery_selection")

# ============================================================
# DECISION VARIABLES
# ============================================================

# number of batteries and chargers
number_of_batteries = model.addVar(vtype=GRB.INTEGER, lb=0, name="number_of_batteries")
number_of_chargers = model.addVar(vtype=GRB.INTEGER, lb=0, name="number_of_chargers")

# battery availibility decisions
at_depot = model.addVars(batteries, times, vtype=GRB.BINARY, name="at_depot")
charging = model.addVars(batteries, times, vtype=GRB.BINARY, name="charging")
discharging = model.addVars(batteries, times, vtype=GRB.BINARY, name="discharging")

# counting number of batteries required
selected = model.addVars(batteries, routes, vtype=GRB.BINARY, name="selected")
used = model.addVars(batteries, vtype=GRB.BINARY, name="used")

# battery energy state
battery_energy = model.addVars(batteries, times, vtype=GRB.CONTINUOUS,
                               lb=MINIMUM_SOC * BATTERY_CAPACITY_KWH,
                               ub=MAXIMUM_SOC * BATTERY_CAPACITY_KWH,
                               name="battery_energy")

# ============================================================
# CONSTRAINTS
# ============================================================

# Each route needs a battery
for r in routes:
    model.addConstr(gp.quicksum(selected[b, r] for b in batteries) == 1, name=f"route_batteries_{r}")

# Charging and discharging over time
for b in batteries:
    model.addConstr(battery_energy[b,0] == INITIAL_SOC * BATTERY_CAPACITY_KWH, name=f"initialise_batteries_{b}")
    for t in times[1:]:
        model.addConstr(discharging[b, t] == gp.quicksum(selected[b, r] for r in active_routes[t]), name=f"discharging_{b}_{t}")

        # If Battery (B) not on a route, it is at depot
        model.addConstr(at_depot[b, t] == 1 - discharging[b, t],name=f"depot_{b}_{t}")
        
        # If Battery (B) is at depot, can be charging (1) or static (0)
        model.addConstr(charging[b, t] <= at_depot[b, t], name=f"charging_or_static_{b}_{t}")
        
        # Battery energy goes up if charging and down if discharging
        model.addConstr(charging[b, t] <= number_of_chargers, name=f"number_batteries_charging_{b}_{t}")
        model.addConstr(battery_energy[b, t] == battery_energy[b, t-1]
                        + CHARGE_POWER_KW_PER_HOUR * CHARGE_EFFICIENCY * charging[b, t]
                        - route_energy_kwh[r] * selected[b, r], 
                        name=f"balance_{b}_{t}")

model.addConstrs((selected[b, r] <= used[b] for b in batteries for r in routes), name="link")
model.addConstr(number_of_batteries == used.sum(), name="count_batteries")

# ============================================================
# OBJECTIVES
# ============================================================


model.setObjectiveN(number_of_batteries, index=0, priority=0, name="MinBatteries")
model.setObjectiveN(number_of_chargers, index=1, priority=1, name="MinChargers")
model.ModelSense = GRB.MINIMIZE

# ============================================================
# SOLVE
# ============================================================

model.optimize()


# ============================================================
# RESULTS
# ============================================================

if model.Status == GRB.INFEASIBLE:
    print("Model is INFEASIBLE")
else:
    print("Model is FEASIBLE")

    rows = []
    for b in batteries:
        for t in times:
            rows.append({
                "battery": b,
                "time": t,
                "charging": charging[b, t].X if t > 0 else 0,
                "discharging": discharging[b, t].X if t > 0 else 0,
                "battery_energy_kwh": battery_energy[b, t].X
            })

    pd.DataFrame(rows).to_csv("battery_charges_over_time.csv", index=False)


