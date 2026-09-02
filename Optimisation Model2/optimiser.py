#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:57:48 2026

"""

"""
DairyDrive Electric - Swappable Battery Optimisation Module


CURRENT Objective
---------
Minimise total cost:
    battery cost x number of selected batteries
    + charger cost x number of selected chargers
    + BESS capex per kWh x installed BESS capacity
    + grid electricity price x total grid import
    - feed-in tariff x total grid/PV export

grid_electricity_price is currently a flat rate. Once the time-of-use
tariff block exists, replace it with a per-timestep dict the same way
pv_generated_t is handled, and update the objective sum accordingly.


Why this file looks different from earlier versions
-----------------------------------------------------
Every constraint family that used to be built with a Python "for b in
batteries: for t in times:" loop (one addConstr call per battery per
hour) is now built as ONE array-level addConstr call using addMVar. This
matters a lot at large horizons: at 365 days (8760 hours) the old loops
were creating tens of thousands of individual constraint objects in
Python before Gurobi even started solving, and that build cost doesn't
show up anywhere in Gurobi's own solve log - it just makes the whole run
feel slow. Vectorizing removes almost all of it.

This does NOT make the underlying combinatorial search easier - the
model still has the same number of binary variables and the same
feasible region. Two things below DO help with that:
  1. A symmetry-breaking constraint on sb_active (see Section 0 below).
     Candidate batteries are otherwise fully interchangeable, so without
     this the solver wastes time exploring relabelings of the same
     solution (e.g. "batteries 1,2,4 active" vs "batteries 1,3,4 active"
     are the same real-world answer). This is the "binary variable
     explosion related to number of batteries" the team flagged.
  2. The bilinear big-M bug fixed in battery_storage_block.py (E_BESS_cap
     x binary was creating quadratic constraints) is still fixed here.

IMPORTANT: this is an algebraic rewrite of the same model, not a change
to what the model represents. Before trusting it at large scale, run it
on a small case (e.g. 5 days) and confirm the objective value and
battery/charger counts match a run from before this rewrite.


Outputs
-------
    - Minimum number of swappable batteries required
    - Minimum number of chargers required
    - Battery state of charge over time
    - Battery-to-route assignments
    - Installed BESS capacity

"""


import numpy as np
import gurobipy as gp
from gurobipy import GRB
import battery_storage_block as bess
import grid_block as grid


def optimiser(routes,times,first_t,prev_t,batteries,active_routes_at,truck_consumption,
    dt,sb_cap,E_SB_min,E_SB_max,E_SB_initial,charge_efficiency,discharge_efficiency, 
    battery_cost,charger_power_kw,charger_cost,max_sb_charge_per_step,soc_BESS_initial,
    pv_generated_t,bess_min_soc,bess_max_soc,bess_capex_per_kwh,
    bess_charge_efficiency,bess_discharge_efficiency,bess_charge_c_rate,bess_discharge_c_rate,
    bess_max_capacity_kwh=1e6,
    grid_electricity_price=0.0, feed_in_tariff=0.0,
    grid_existing_kw=0.0, grid_max_kw=1e6,
    grid_import_max=None, grid_export_max=None,
    time_limit=None):

    # Defaults so this still runs before main.py supplies real grid sizing data.
    # Replace with real values from input_data.csv as soon as they're available.
    if grid_import_max is None:
        grid_import_max = {t: 1e6 for t in times}
    if grid_export_max is None:
        grid_export_max = {t: 1e6 for t in times}

    n_b = len(batteries)
    n_t = len(times)
    n_r = len(routes)

    # Times must be contiguous and in order for the shifted-array SOC
    # recursion below to be equivalent to the original per-t "previous
    # timestep" lookup. This matches how main.py builds TIMES/PREV_T.
    for i in range(1, n_t):
        assert prev_t[times[i]] == times[i - 1], (
            "optimiser() assumes contiguous, ordered timesteps"
        )


    # ============================================================
    # Model
    # ============================================================
    
    m = gp.Model("battery_selection")

    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)
    
    # ============================================================
    # Decision Variables
    # ============================================================
    
    # Batteries and Chargers Variables
    sb_active = m.addMVar(n_b, vtype=GRB.BINARY, name="battery_active")
    N_CS = m.addVar(vtype=GRB.INTEGER, lb=0, ub=n_b, name="number_of_chargers")
    
    # Battery Depot, Route & Charging, Discharging State Variables
    # assign_SB | sb_active
    # 0 | 1 : At depot, charging
    # 0 | 0 : At depot, Idle
    # 1 | 1/0: On route --> Discharging
    assign_SB = m.addMVar((n_r, n_b), vtype=GRB.BINARY, name="on_route")
    x_SB_charge = m.addMVar((n_b, n_t), vtype=GRB.BINARY, name="charging")
    
    # Battery Energy (Charge Amount and Energy Level) (kWh) 
    f_SB = m.addMVar((n_b, n_t), lb=0, ub=max_sb_charge_per_step, name="charge_energy_amount")
    E_SB = m.addMVar((n_b, n_t), lb=0, ub=E_SB_max, name="battery_energy_level")
    
    # Grid Import and Site Export Variables
    grid_import = m.addMVar(n_t, lb=0, name="Grid_import")
    pv_export = m.addMVar(n_t, lb=0, name="PV_export")


    # ============================================================
    # 0. Symmetry breaking
    # ============================================================
    # Candidate batteries are interchangeable, so without this the
    # solver wastes time exploring relabelings of the same real
    # solution. Forcing batteries to be "switched on" in index order
    # removes that redundancy without changing the optimal battery
    # count or any other result.
    m.addConstr(sb_active[:-1] >= sb_active[1:], name="SB_SymmetryBreak")
    
    
    # ============================================================
    # 1. Route assignment constraints (DECLAN)
    # ============================================================
    
    # Every route needs exactly one battery
    m.addConstr(assign_SB.sum(axis=1) == 1, name="Route_OneBattery")
    # A route can only be assigned to a purchased battery.
    # assign_SB is (n_r, n_b); sb_active is (n_b,) - this broadcasts
    # sb_active across the route axis, which is the normal (rightmost
    # axis matches) numpy broadcasting case, so no reshape is needed.
    m.addConstr(assign_SB <= sb_active, name="Route_ActiveBattery")
    
    
    # ============================================================
    # 2. Battery availability, SOC, route-energy, and charging
    #    constraints (DECLAN)
    #
    #    away[b,t] (whether battery b is out on a route at time t) is
    #    built once as an (n_b, n_t) array via a route/time incidence
    #    matrix, instead of rebuilding a quicksum per (b,t) pair.
    # ============================================================

    # Constant route/time incidence matrix: active_at[r_idx, t_idx] = 1
    # if route routes[r_idx] is active at times[t_idx].
    route_idx = {r: i for i, r in enumerate(routes)}
    active_at = np.zeros((n_r, n_t))
    for t_idx, t in enumerate(times):
        for r in active_routes_at[t]:
            active_at[route_idx[r], t_idx] = 1

    # away[b, t] = sum_r assign_SB[r, b] * active_at[r, t]
    away = assign_SB.T @ active_at  # shape (n_b, n_t)

    # sb_active broadcast across the time axis. This is the OTHER
    # broadcasting direction from Route_ActiveBattery above: sb_active's
    # single axis needs to line up with axis 0 (batteries) of an
    # (n_b, n_t) array, not the trailing axis, so numpy's normal
    # broadcasting won't do it automatically - build it explicitly via
    # a constant-times-MVar matrix product instead.
    sb_active_bt = sb_active.reshape((n_b, 1)) @ np.ones((1, n_t))

    # Battery availability: can't be on two routes at once, and can't
    # charge while away on a route.
    m.addConstr(away <= sb_active_bt, name="SB_NoRouteOverlap")
    m.addConstr(x_SB_charge + away <= sb_active_bt, name="SB_NoChargeWhileAway")

    # Storage bounds switch off automatically when the candidate battery
    # is not purchased.
    m.addConstr(E_SB >= E_SB_min * sb_active_bt, name="SB_SOCMin")
    m.addConstr(E_SB <= E_SB_max * sb_active_bt, name="SB_SOCMax")

    # Charging is possible only for an active battery and one charger slot.
    m.addConstr(f_SB <= max_sb_charge_per_step * x_SB_charge, name="SB_ChargeRate")

    # Constant route energy is consumed during every hour the battery is on a route.
    route_energy_used = truck_consumption * dt * away

    # SOC state equation, written as two array constraints (first hour,
    # then every other hour at once) instead of one constraint per hour.
    # If route_energy_kwh is already battery-side consumption, set
    # sb_discharge_efficiency = 1.0 in the input data.
    m.addConstr(
        E_SB[:, 0] == E_SB_initial * sb_active
        + charge_efficiency * f_SB[:, 0]
        - route_energy_used[:, 0] / discharge_efficiency,
        name="SB_EnergyBalance_first"
    )
    m.addConstr(
        E_SB[:, 1:] == E_SB[:, :-1]
        + charge_efficiency * f_SB[:, 1:]
        - route_energy_used[:, 1:] / discharge_efficiency,
        name="SB_EnergyBalance_rest"
    )

    # Each battery charging uses 1 charger (sum over batteries, at each timestep)
    m.addConstr(x_SB_charge.sum(axis=0) <= N_CS, name="Charger_Count")
    m.addConstr(f_SB.sum(axis=0) <= N_CS * charger_power_kw * dt, name="Charger_TotalEnergy")
    
    
    # ============================================================
    # 3. Stationary Battery Constraints
    # ============================================================

    sb_charging_energy = f_SB.sum(axis=0)  # length n_t

    bess_vars = bess.build_bess_block(
        model=m,
        times=times,
        first_t=first_t,
        prev_t=prev_t,
        dt=dt,
        soc_min=bess_min_soc,
        soc_max=bess_max_soc,
        starting_soc=soc_BESS_initial,
        charge_efficiency=bess_charge_efficiency,
        discharge_efficiency=bess_discharge_efficiency,
        charge_c_rate=bess_charge_c_rate,
        discharge_c_rate=bess_discharge_c_rate,
        grid_import=grid_import,
        pv_generated_t=pv_generated_t,
        pv_export=pv_export,
        sb_charging_energy=sb_charging_energy,
        max_capacity_kwh=bess_max_capacity_kwh,
    )

    E_BESS_cap = bess_vars["E_BESS_cap"]
    E_BESS = bess_vars["E_BESS"]
    f_BESS = bess_vars["f_BESS"]
    p_BESS = bess_vars["p_BESS"]
    x_BESS = bess_vars["x_BESS"]


    # ============================================================
    # 4. Grid Connection Constraints
    # ============================================================

    grid_vars = grid.build_grid_block(
        model=m,
        times=times,
        dt=dt,
        grid_import=grid_import,
        grid_export=pv_export,
        grid_existing_kw=grid_existing_kw,
        grid_max_kw=grid_max_kw,
        grid_import_max=grid_import_max,
        grid_export_max=grid_export_max,
    )

    S_g = grid_vars["S_g"]
    x_grid_import = grid_vars["x_grid_import"]
    
    
    # ============================================================
    # Objective Function Definition
    # ============================================================
    
    m.setObjective(
        battery_cost * sb_active.sum()
        + charger_cost * N_CS
        + bess_capex_per_kwh * E_BESS_cap
        + grid_electricity_price * dt * grid_import.sum()
        - feed_in_tariff * dt * pv_export.sum(),
        GRB.MINIMIZE
    )
    
    m.optimize()
    
    


    # ============================================================
    # Visualise Results
    # ============================================================
    
    import matplotlib.pyplot as plt
    
    if m.status == GRB.OPTIMAL:

        print(E_BESS_cap.X)

        n_batteries = sb_active.X.sum()
        n_chargers = N_CS.X
        print(f"Batteries needed: {n_batteries:.0f}")
        print(f"Chargers needed:  {n_chargers:.0f}")
    
    
        # active_batteries = [i for i in range(n_b) if sb_active.X[i] > 0.5]
    
        # plt.figure(figsize=(14, 7))
        
        # for i in active_batteries:
        #     battery_soc = (E_SB.X[i, :] / sb_cap) * 100
        #     plt.plot(times, battery_soc, label=f"Battery {batteries[i]}")
        
        # plt.xlabel("Time")
        # plt.ylabel("Battery State of Charge (%)")
        # plt.title("Swappable Battery State of Charge Over Time")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

    return