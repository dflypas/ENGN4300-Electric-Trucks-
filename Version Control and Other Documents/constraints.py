"""DairyDrive Electric - MILP constraint module (Gurobi / gurobipy).

This file formulates the constraint set for the DairyDrive Electric optimisation
model. It intentionally does NOT define the cost or GHG objective; the returned
model and variables can be passed to the objective-function module.

Core modelling assumptions used here
------------------------------------
1. Route/truck schedules are INPUTS, not optimisation decisions.
2. A converted truck uses exactly one swappable battery on each route.
3. A swappable battery cannot be on two routes at once and cannot charge while
   it is away on a route.
4. Route energy is withdrawn from the assigned battery at the route departure
   timestep. This is an event-based bookkeeping convention; the battery remains
   unavailable for the entire route duration.
5. Swappable-battery charging happens at the depot/station. Route traction
   energy is therefore NOT also included as a depot electrical load, avoiding
   double-counting of energy.
6. Stationary BESS capacity is a continuous decision variable in kWh, whereas
   swappable batteries are counted as individual standard units.
7. Grid import/export are mutually exclusive at each timestep.
8. PV generation is determined by installed panel count and an input generation
   profile; curtailment is represented explicitly.
9. Exact numerical SOC buffers, swap duration, forklift limits, and replacement
   downtime are not assumed here because they were not supplied. SOC limits and
   efficiencies are input parameters.

Expected data dictionary
------------------------
Required sets / mappings:
    trucks                       iterable of truck IDs
    years                        ordered iterable of project years (e.g. range(10))
    time                         ordered iterable of hourly timestep IDs
    routes                       iterable of route-event IDs
    route_truck[r]               truck assigned to route r (input schedule)
    route_year[r]                project year containing route r
    route_departure_t[r]         timestep at which route r departs
    route_active[(r,t)]          1 while route r occupies its truck/battery, else 0
    route_energy_kwh[r]          traction energy required by route r (kWh delivered)

Swappable battery / charger parameters:
    max_swappable_batteries      candidate upper bound used to build the MILP
    sb_capacity_kwh
    sb_soc_min                   fraction [0,1]
    sb_soc_max                   fraction [0,1]
    sb_soc_initial               fraction [0,1]
    sb_charge_efficiency         fraction (0,1]
    sb_discharge_efficiency      fraction (0,1]; set to 1.0 if route_energy_kwh is
                                 already measured as battery-side energy consumption
    sb_max_charge_kw
    charger_power_kw
    max_chargers

PV parameters:
    pv_panel_area_m2
    available_pv_area_m2
    pv_kwh_per_panel[t]          available PV energy per panel in timestep t (kWh)

Grid parameters:
    existing_grid_capacity_kw
    max_grid_connection_kw       hard upper bound / maximum feasible connection size
    max_grid_import_kw[t]        utility/import hard limit at t
    max_grid_export_kw[t]        utility/export hard limit at t

Stationary BESS parameters:
    max_bess_capacity_kwh
    bess_soc_min                 fraction [0,1]
    bess_soc_max                 fraction [0,1]
    bess_soc_initial             fraction [0,1]
    bess_charge_efficiency       fraction (0,1]
    bess_discharge_efficiency    fraction (0,1]
    bess_charge_c_rate_per_h     e.g. 0.5 means max charge power = 0.5*capacity kW
    bess_discharge_c_rate_per_h

Other:
    dt_hours                     default 1.0
    aux_load_kwh[t]              optional depot auxiliary load; default 0
    initial_ev[i]                optional 0/1 initial conversion state; default 0
    cyclic_storage               optional bool; if True, end SOC equals initial SOC.
                                 If False (default), end SOC must be at least initial.

All energy-flow variables (u, v, f, p, E_PV) are in kWh per timestep.
All storage-state variables are in kWh.
"""

import gurobipy as gp
from gurobipy import GRB


def _validate_fraction(name, value, allow_zero=True):
    lo = 0.0 if allow_zero else 0.0
    if value < lo or value > 1.0 or (not allow_zero and value == 0.0):
        raise ValueError(f"{name} must be in {'(0, 1]' if not allow_zero else '[0, 1]'}. Got {value}.")


def build_dairydrive_constraint_model(data):
    """Build and return the Gurobi model and a dictionary of decision variables."""

    # -----------------------------
    # Sets and ordered index lists
    # -----------------------------
    I = list(data["trucks"])
    Y = list(data["years"])
    T = list(data["time"])
    R = list(data["routes"])

    if not I or not Y or not T:
        raise ValueError("trucks, years, and time must all be non-empty.")

    B = list(range(int(data["max_swappable_batteries"])))
    if not B:
        raise ValueError("max_swappable_batteries must be at least 1.")

    dt = float(data.get("dt_hours", 1.0))
    if dt <= 0:
        raise ValueError("dt_hours must be > 0.")

    prev_y = {Y[k]: Y[k - 1] for k in range(1, len(Y))}
    prev_t = {T[k]: T[k - 1] for k in range(1, len(T))}
    first_y = Y[0]
    first_t = T[0]
    last_t = T[-1]

    route_truck = data["route_truck"]
    route_year = data["route_year"]
    route_departure = data["route_departure_t"]
    route_active = data["route_active"]
    route_energy = data["route_energy_kwh"]

    # Validate route references and the fixed truck schedule.
    I_set, Y_set, T_set = set(I), set(Y), set(T)
    for r in R:
        if route_truck[r] not in I_set:
            raise ValueError(f"Route {r} references unknown truck {route_truck[r]}.")
        if route_year[r] not in Y_set:
            raise ValueError(f"Route {r} references unknown year {route_year[r]}.")
        if route_departure[r] not in T_set:
            raise ValueError(f"Route {r} has departure timestep outside T.")
        if route_energy[r] < 0:
            raise ValueError(f"Route {r} has negative route energy.")

    # A fixed route/truck schedule must itself be feasible: one truck cannot be
    # on two input routes simultaneously. This is a data validation, not a MILP decision.
    for i in I:
        for t in T:
            overlap = sum(int(route_active.get((r, t), 0)) for r in R if route_truck[r] == i)
            if overlap > 1:
                raise ValueError(
                    f"Input schedule is infeasible: truck {i} is assigned to {overlap} routes at time {t}."
                )

    # -----------------------------
    # Scalar parameters
    # -----------------------------
    F = len(I)

    sb_cap = float(data["sb_capacity_kwh"])
    sb_soc_min = float(data["sb_soc_min"])
    sb_soc_max = float(data["sb_soc_max"])
    sb_soc_initial = float(data["sb_soc_initial"])
    eta_sb_ch = float(data["sb_charge_efficiency"])
    eta_sb_dis = float(data["sb_discharge_efficiency"])
    sb_max_charge_kw = float(data["sb_max_charge_kw"])

    _validate_fraction("sb_soc_min", sb_soc_min)
    _validate_fraction("sb_soc_max", sb_soc_max)
    _validate_fraction("sb_soc_initial", sb_soc_initial)
    _validate_fraction("sb_charge_efficiency", eta_sb_ch, allow_zero=False)
    _validate_fraction("sb_discharge_efficiency", eta_sb_dis, allow_zero=False)
    if not (sb_soc_min <= sb_soc_initial <= sb_soc_max):
        raise ValueError("sb_soc_initial must lie between sb_soc_min and sb_soc_max.")

    charger_power_kw = float(data["charger_power_kw"])
    max_chargers = int(data["max_chargers"])

    panel_area = float(data["pv_panel_area_m2"])
    pv_area = float(data["available_pv_area_m2"])
    pv_per_panel = data["pv_kwh_per_panel"]

    grid_existing_kw = float(data["existing_grid_capacity_kw"])
    grid_max_kw = float(data["max_grid_connection_kw"])
    grid_import_max = data["max_grid_import_kw"]
    grid_export_max = data["max_grid_export_kw"]
    if grid_existing_kw < 0 or grid_max_kw < grid_existing_kw:
        raise ValueError("Grid capacity bounds are invalid.")

    bess_max_cap = float(data["max_bess_capacity_kwh"])
    bess_soc_min = float(data["bess_soc_min"])
    bess_soc_max = float(data["bess_soc_max"])
    bess_soc_initial = float(data["bess_soc_initial"])
    eta_bess_ch = float(data["bess_charge_efficiency"])
    eta_bess_dis = float(data["bess_discharge_efficiency"])
    bess_c_ch = float(data["bess_charge_c_rate_per_h"])
    bess_c_dis = float(data["bess_discharge_c_rate_per_h"])

    _validate_fraction("bess_soc_min", bess_soc_min)
    _validate_fraction("bess_soc_max", bess_soc_max)
    _validate_fraction("bess_soc_initial", bess_soc_initial)
    _validate_fraction("bess_charge_efficiency", eta_bess_ch, allow_zero=False)
    _validate_fraction("bess_discharge_efficiency", eta_bess_dis, allow_zero=False)
    if not (bess_soc_min <= bess_soc_initial <= bess_soc_max):
        raise ValueError("bess_soc_initial must lie between bess_soc_min and bess_soc_max.")

    aux_load = data.get("aux_load_kwh", {t: 0.0 for t in T})
    initial_ev = data.get("initial_ev", {i: 0 for i in I})
    cyclic_storage = bool(data.get("cyclic_storage", False))

    # -----------------------------
    # Model
    # -----------------------------
    m = gp.Model("DairyDrive_Electric_Constraints")

    # -----------------------------
    # Fleet conversion variables
    # -----------------------------
    # is_ev[i,y] = 1 if truck i has been converted by year y.
    is_ev = m.addVars(I, Y, vtype=GRB.BINARY, name="isEV")
    converted = m.addVars(I, Y, vtype=GRB.BINARY, name="convert")
    N_EV = m.addVars(Y, lb=0, ub=F, vtype=GRB.INTEGER, name="N_EV")
    N_D = m.addVars(Y, lb=0, ub=F, vtype=GRB.INTEGER, name="N_Diesel")

    # -----------------------------
    # Infrastructure variables
    # -----------------------------
    max_panels = int(pv_area // panel_area) if panel_area > 0 else 0
    N_PV = m.addVar(lb=0, ub=max_panels, vtype=GRB.INTEGER, name="N_PV")
    N_CS = m.addVar(lb=0, ub=max_chargers, vtype=GRB.INTEGER, name="N_Charger")

    sb_active = m.addVars(B, vtype=GRB.BINARY, name="SB_active")
    N_SB = m.addVar(lb=0, ub=len(B), vtype=GRB.INTEGER, name="N_SB")

    # NBESS is installed stationary storage capacity (kWh), not a count of units.
    N_BESS = m.addVar(lb=0, ub=bess_max_cap, vtype=GRB.CONTINUOUS, name="BESS_kWh")

    # S_g is installed bidirectional grid-connection capacity (kW).
    S_g = m.addVar(
        lb=grid_existing_kw,
        ub=grid_max_kw,
        vtype=GRB.CONTINUOUS,
        name="GridConnection_kW",
    )

    # -----------------------------
    # Grid and PV variables
    # -----------------------------
    u_g = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="GridImport_kWh")
    v_g = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="GridExport_kWh")
    x_g = m.addVars(T, vtype=GRB.BINARY, name="GridImportMode")

    E_PV = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="PVGeneration_kWh")
    E_PV_curt = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="PVCurtailed_kWh")

    # -----------------------------
    # Swappable battery variables
    # -----------------------------
    E_SB = m.addVars(B, T, lb=0, vtype=GRB.CONTINUOUS, name="SB_Energy_kWh")
    f_SB = m.addVars(B, T, lb=0, vtype=GRB.CONTINUOUS, name="SB_ChargeIn_kWh")
    p_SB = m.addVars(B, T, lb=0, vtype=GRB.CONTINUOUS, name="SB_RouteOut_kWh")
    x_SB_charge = m.addVars(B, T, vtype=GRB.BINARY, name="SB_Charging")

    # assign_SB[r,b] = 1 if battery b is assigned to route event r.
    assign_SB = m.addVars(R, B, vtype=GRB.BINARY, name="SB_Assign")

    # -----------------------------
    # Stationary BESS variables
    # -----------------------------
    E_BESS = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="BESS_Energy_kWh")
    f_BESS = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="BESS_ChargeIn_kWh")
    p_BESS = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS, name="BESS_DischargeOut_kWh")
    x_BESS = m.addVars(T, vtype=GRB.BINARY, name="BESS_ChargeMode")

    # ============================================================
    # 1. Fleet constraints
    # ============================================================
    for y in Y:
        m.addConstr(N_EV[y] == gp.quicksum(is_ev[i, y] for i in I), name=f"Fleet_EVCount[{y}]")
        m.addConstr(N_D[y] + N_EV[y] == F, name=f"Fleet_Total[{y}]")

    for i in I:
        init = int(initial_ev.get(i, 0))
        if init not in (0, 1):
            raise ValueError(f"initial_ev[{i}] must be 0 or 1.")

        # First project year: conversion is the change from the initial state.
        if init == 0:
            m.addConstr(converted[i, first_y] == is_ev[i, first_y], name=f"ConvertFirst[{i}]")
        else:
            m.addConstr(is_ev[i, first_y] == 1, name=f"InitiallyEV[{i}]")
            m.addConstr(converted[i, first_y] == 0, name=f"NoFirstConversion[{i}]")

        # Once converted, a truck remains electric.
        for y in Y[1:]:
            py = prev_y[y]
            m.addConstr(is_ev[i, y] >= is_ev[i, py], name=f"ConversionMonotonic[{i},{y}]")
            m.addConstr(converted[i, y] == is_ev[i, y] - is_ev[i, py], name=f"ConversionEvent[{i},{y}]")

    # ============================================================
    # 2. Infrastructure count/capacity constraints
    # ============================================================
    m.addConstr(N_SB == gp.quicksum(sb_active[b] for b in B), name="SB_Count")

    # At least one physical swappable battery per converted truck.
    for y in Y:
        m.addConstr(N_SB >= N_EV[y], name=f"SB_MinForEVFleet[{y}]")

    # PV roof/land area limit.
    m.addConstr(N_PV * panel_area <= pv_area, name="PV_AreaLimit")

    # ============================================================
    # 3. Route assignment and battery availability constraints
    # ============================================================
    for r in R:
        i = route_truck[r]
        y = route_year[r]

        # Diesel route => no swappable battery assignment.
        # Electric route => exactly one swappable battery assignment.
        m.addConstr(
            gp.quicksum(assign_SB[r, b] for b in B) == is_ev[i, y],
            name=f"Route_OneBatteryIfEV[{r}]",
        )

        for b in B:
            m.addConstr(assign_SB[r, b] <= sb_active[b], name=f"Route_ActiveBattery[{r},{b}]")

    # A battery cannot be on more than one route simultaneously, and it cannot
    # charge while it is away on a route.
    for b in B:
        for t in T:
            away_expr = gp.quicksum(
                assign_SB[r, b] * int(route_active.get((r, t), 0)) for r in R
            )
            m.addConstr(away_expr <= sb_active[b], name=f"SB_NoRouteOverlap[{b},{t}]")
            m.addConstr(
                x_SB_charge[b, t] + away_expr <= sb_active[b],
                name=f"SB_NoChargeWhileAway[{b},{t}]",
            )

    # ============================================================
    # 4. Swappable-battery SOC, route-energy, and charging constraints
    # ============================================================
    E_SB_min = sb_soc_min * sb_cap
    E_SB_max = sb_soc_max * sb_cap
    E_SB_initial = sb_soc_initial * sb_cap
    max_sb_charge_per_step = min(sb_max_charge_kw, charger_power_kw) * dt

    routes_departing_at = {t: [r for r in R if route_departure[r] == t] for t in T}

    for b in B:
        for t in T:
            # Storage bounds switch off automatically when the candidate battery is not purchased.
            m.addConstr(E_SB[b, t] >= E_SB_min * sb_active[b], name=f"SB_SOCMin[{b},{t}]")
            m.addConstr(E_SB[b, t] <= E_SB_max * sb_active[b], name=f"SB_SOCMax[{b},{t}]")

            # Charging is possible only for an active battery and one charger slot.
            m.addConstr(x_SB_charge[b, t] <= sb_active[b], name=f"SB_ChargeOnlyIfActive[{b},{t}]")
            m.addConstr(
                f_SB[b, t] <= max_sb_charge_per_step * x_SB_charge[b, t],
                name=f"SB_ChargeRate[{b},{t}]",
            )

            # Route-energy output occurs at the route departure event.
            m.addConstr(
                p_SB[b, t]
                == gp.quicksum(route_energy[r] * assign_SB[r, b] for r in routes_departing_at[t]),
                name=f"SB_RouteEnergyOut[{b},{t}]",
            )

            previous_energy = E_SB_initial * sb_active[b] if t == first_t else E_SB[b, prev_t[t]]

            # SOC state equation. If route_energy_kwh is already battery-side
            # consumption, set sb_discharge_efficiency = 1.0 in the input data.
            m.addConstr(
                E_SB[b, t]
                == previous_energy
                + eta_sb_ch * f_SB[b, t]
                - p_SB[b, t] / eta_sb_dis,
                name=f"SB_EnergyBalance[{b},{t}]",
            )

        # End-of-horizon condition prevents the optimiser from obtaining a free
        # benefit by emptying batteries at the end of the model horizon.
        if cyclic_storage:
            m.addConstr(
                E_SB[b, last_t] == E_SB_initial * sb_active[b],
                name=f"SB_TerminalSOC[{b}]",
            )
        else:
            m.addConstr(
                E_SB[b, last_t] >= E_SB_initial * sb_active[b],
                name=f"SB_TerminalSOC[{b}]",
            )

    # Explicit pre-departure energy requirement. This is stronger/more readable
    # than relying only on the post-departure SOC lower bound.
    for r in R:
        dep = route_departure[r]
        required_battery_draw = route_energy[r] / eta_sb_dis
        for b in B:
            energy_before_departure = (
                E_SB_initial * sb_active[b] if dep == first_t else E_SB[b, prev_t[dep]]
            )
            m.addConstr(
                energy_before_departure
                >= E_SB_min * sb_active[b] + required_battery_draw * assign_SB[r, b],
                name=f"SB_EnoughEnergyBeforeRoute[{r},{b}]",
            )

    # Charger-count limits. Each charging battery occupies one charger.
    for t in T:
        m.addConstr(
            gp.quicksum(x_SB_charge[b, t] for b in B) <= N_CS,
            name=f"Charger_Count[{t}]",
        )
        m.addConstr(
            gp.quicksum(f_SB[b, t] for b in B) <= N_CS * charger_power_kw * dt,
            name=f"Charger_TotalEnergy[{t}]",
        )

    # ============================================================
    # 5. Stationary BESS constraints
    # ============================================================
    bess_bigM_charge = bess_max_cap * bess_c_ch * dt
    bess_bigM_discharge = bess_max_cap * bess_c_dis * dt

    for t in T:
        m.addConstr(E_BESS[t] >= bess_soc_min * N_BESS, name=f"BESS_SOCMin[{t}]")
        m.addConstr(E_BESS[t] <= bess_soc_max * N_BESS, name=f"BESS_SOCMax[{t}]")

        # Power-to-energy (C-rate) limits scale with installed BESS capacity.
        m.addConstr(
            f_BESS[t] <= bess_c_ch * N_BESS * dt,
            name=f"BESS_ChargeRateCapacity[{t}]",
        )
        m.addConstr(
            p_BESS[t] <= bess_c_dis * N_BESS * dt,
            name=f"BESS_DischargeRateCapacity[{t}]",
        )

        # Binary mode prevents simultaneous BESS charge and discharge.
        m.addConstr(
            f_BESS[t] <= bess_bigM_charge * x_BESS[t],
            name=f"BESS_ChargeMode[{t}]",
        )
        m.addConstr(
            p_BESS[t] <= bess_bigM_discharge * (1 - x_BESS[t]),
            name=f"BESS_DischargeMode[{t}]",
        )

        previous_bess_energy = bess_soc_initial * N_BESS if t == first_t else E_BESS[prev_t[t]]
        m.addConstr(
            E_BESS[t]
            == previous_bess_energy
            + eta_bess_ch * f_BESS[t]
            - p_BESS[t] / eta_bess_dis,
            name=f"BESS_EnergyBalance[{t}]",
        )

    if cyclic_storage:
        m.addConstr(E_BESS[last_t] == bess_soc_initial * N_BESS, name="BESS_TerminalSOC")
    else:
        m.addConstr(E_BESS[last_t] >= bess_soc_initial * N_BESS, name="BESS_TerminalSOC")

    # ============================================================
    # 6. PV constraints
    # ============================================================
    for t in T:
        if pv_per_panel[t] < 0:
            raise ValueError(f"pv_kwh_per_panel[{t}] cannot be negative.")

        # Available generation is fixed by installed panel count and the input profile.
        m.addConstr(E_PV[t] == N_PV * pv_per_panel[t], name=f"PV_Generation[{t}]")
        m.addConstr(E_PV_curt[t] <= E_PV[t], name=f"PV_CurtailmentLimit[{t}]")

    # ============================================================
    # 7. Grid import/export constraints
    # ============================================================
    for t in T:
        M_imp = float(grid_import_max[t])
        M_exp = float(grid_export_max[t])
        if M_imp < 0 or M_exp < 0:
            raise ValueError(f"Grid limits at time {t} cannot be negative.")

        # Installed connection capacity.
        m.addConstr(u_g[t] <= S_g * dt, name=f"Grid_ConnectionImport[{t}]")
        m.addConstr(v_g[t] <= S_g * dt, name=f"Grid_ConnectionExport[{t}]")

        # Utility hard limits + mutually exclusive import/export mode.
        m.addConstr(u_g[t] <= M_imp * dt * x_g[t], name=f"Grid_ImportMode[{t}]")
        m.addConstr(v_g[t] <= M_exp * dt * (1 - x_g[t]), name=f"Grid_ExportMode[{t}]")

    # ============================================================
    # 8. Site electrical energy balance
    # ============================================================
    for t in T:
        # IMPORTANT: p_SB is traction energy delivered away from the depot and is
        # not added here. That energy was previously supplied to the battery via f_SB.
        m.addConstr(
            u_g[t] + E_PV[t] + p_BESS[t]
            == gp.quicksum(f_SB[b, t] for b in B)
            + f_BESS[t]
            + v_g[t]
            + E_PV_curt[t]
            + float(aux_load.get(t, 0.0)),
            name=f"Site_EnergyBalance[{t}]",
        )

    # No economic or emissions objective is set here. A zero objective makes the
    # model solvable as a pure feasibility model for debugging.
    m.setObjective(0.0, GRB.MINIMIZE)

    variables = {
        "is_ev": is_ev,
        "converted": converted,
        "N_EV": N_EV,
        "N_Diesel": N_D,
        "N_PV": N_PV,
        "N_Charger": N_CS,
        "SB_active": sb_active,
        "N_SB": N_SB,
        "N_BESS_kWh": N_BESS,
        "GridConnection_kW": S_g,
        "u_g": u_g,
        "v_g": v_g,
        "x_g": x_g,
        "E_PV": E_PV,
        "E_PV_curtailed": E_PV_curt,
        "E_SB": E_SB,
        "f_SB": f_SB,
        "p_SB": p_SB,
        "x_SB_charge": x_SB_charge,
        "BatteryAssignment": assign_SB,
        "E_BESS": E_BESS,
        "f_BESS": f_BESS,
        "p_BESS": p_BESS,
        "x_BESS": x_BESS,
    }

    return m, variables
