"""
battery_storage_block.py

Adds the stationary BESS part to the shared model. This is the single
authoritative BESS implementation - optimiser.py calls this instead of
building BESS variables/constraints inline.

Matches Section 8 (Stationary BESS, B1-B8) of the constraints document -
same B1-B8 as before, just rewritten with addMVar so each constraint
family is built as ONE array-level addConstr call instead of one call per
timestep. At 365 days (8760 hours) the old per-timestep loops were
building ~9 x 8760 = ~79,000 separate constraint objects in Python before
Gurobi even started solving; that Python-side build cost doesn't show up
in Gurobi's own log at all, but it dominates wall-clock time at this
scale. Vectorizing removes it almost entirely.

The temporal SOC recursion (B7) can't be fully vectorized away - each
hour genuinely depends on the previous hour - but it CAN be written as
two array constraints (one for the first hour, one for every other hour
at once) instead of one constraint per hour, by comparing the energy
array against a shifted copy of itself. This assumes times are
contiguous and in order (true given how main.py builds TIMES/PREV_T).

Linearity note (unchanged from before): B4/B5's mode-gated big-M uses a
FIXED constant, not E_BESS_cap, so charge/discharge mode gating stays
linear instead of creating a bilinear (E_BESS_cap x binary) term.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def build_bess_block(model, times, first_t, prev_t, dt,
                      soc_min, soc_max, starting_soc,
                      charge_efficiency, discharge_efficiency,
                      charge_c_rate, discharge_c_rate,
                      grid_import, pv_generated_t, pv_export,
                      sb_charging_energy, max_capacity_kwh=1e6,
                      max_charge_power_kw=None, max_discharge_power_kw=None):

    n_t = len(times)

    # Times must be contiguous and in order for the shifted-array SOC
    # recursion below to be equivalent to the original per-t "previous
    # timestep" lookup. This matches how main.py builds TIMES/PREV_T.
    for i in range(1, n_t):
        assert prev_t[times[i]] == times[i - 1], (
            "build_bess_block assumes contiguous, ordered timesteps"
        )

    # pv_generated_t arrives as a {t: value} dict from main.py; convert
    # once to a plain array in time order so it can be used in vectorized
    # arithmetic against the MVar arrays below.
    pv_array = np.array([pv_generated_t[t] for t in times])

    if max_charge_power_kw is None:
        max_charge_power_kw = charge_c_rate * max_capacity_kwh
    if max_discharge_power_kw is None:
        max_discharge_power_kw = discharge_c_rate * max_capacity_kwh

    # B1: installed energy capacity in kWh, bounded above by a
    # feasibility limit so the solver can't install unlimited capacity.
    E_BESS_cap = model.addVar(lb=0, ub=max_capacity_kwh, name="Capacity_BESS_kWh")

    # B2: installed charge/discharge power capacity, separate decision
    # variables from energy capacity, each with its own feasibility limit.
    P_BESS_ch_inst = model.addVar(lb=0, ub=max_charge_power_kw, name="ChargePower_Installed_kW")
    P_BESS_dis_inst = model.addVar(lb=0, ub=max_discharge_power_kw, name="DischargePower_Installed_kW")

    # energy stored in the BESS each hour
    E_BESS = model.addMVar(n_t, lb=0, name="Energy_BESS_kWh")

    # gross charge/discharge power each hour (before efficiency losses)
    f_BESS = model.addMVar(n_t, lb=0, name="BESS_ChargeIn_kWh")
    p_BESS = model.addMVar(n_t, lb=0, name="BESS_DischargeOut_kWh")

    # 1 = charging this hour, 0 = discharging this hour
    x_BESS = model.addMVar(n_t, vtype=GRB.BINARY, name="BESS_ChargeMode")

    # B3: SOC stays within min/max of installed capacity.
    # E_BESS_cap is a plain scalar Var here, so soc_min * E_BESS_cap is a
    # scalar expression that broadcasts against the length-n_t array -
    # this is a normal scalar-vs-array broadcast, not the axis-mismatch
    # case that needs an explicit reshape (see the (n,t) shaped
    # constraints in optimiser.py for that case).
    model.addConstr(E_BESS >= soc_min * E_BESS_cap, name="BESS_SOCMin")
    model.addConstr(E_BESS <= soc_max * E_BESS_cap, name="BESS_SOCMax")

    # B4: charging power limits - installed capacity, fixed-power mode
    # gate (kept linear), and C-rate of installed energy capacity.
    model.addConstr(f_BESS <= P_BESS_ch_inst * dt, name="BESS_ChargeInstalledLimit")
    model.addConstr(f_BESS <= max_charge_power_kw * dt * x_BESS, name="BESS_ChargeMode")
    model.addConstr(f_BESS <= charge_c_rate * E_BESS_cap * dt, name="BESS_ChargeRate")

    # B5: discharging power limits, mirroring B4.
    model.addConstr(p_BESS <= P_BESS_dis_inst * dt, name="BESS_DischargeInstalledLimit")
    model.addConstr(p_BESS <= max_discharge_power_kw * dt * (1 - x_BESS), name="BESS_DischargeMode")
    model.addConstr(p_BESS <= discharge_c_rate * E_BESS_cap * dt, name="BESS_DischargeRate")

    # gross charge/discharge power ties back to available site energy
    model.addConstr(
        f_BESS - p_BESS == grid_import + pv_array - pv_export - sb_charging_energy,
        name="BESS_SiteLink"
    )

    # B7: SOC balance, with efficiency losses applied. Written as two
    # array constraints instead of one per hour: the first hour compares
    # against the starting SOC, every other hour compares against the
    # previous element of the E_BESS array itself.
    model.addConstr(
        E_BESS[0] == starting_soc * E_BESS_cap
        + charge_efficiency * f_BESS[0]
        - p_BESS[0] / discharge_efficiency,
        name="BESS_EnergyBalance_first"
    )
    model.addConstr(
        E_BESS[1:] == E_BESS[:-1]
        + charge_efficiency * f_BESS[1:]
        - p_BESS[1:] / discharge_efficiency,
        name="BESS_EnergyBalance_rest"
    )

    # B8: don't let the optimiser get a free benefit by emptying the BESS at the end
    model.addConstr(E_BESS[n_t - 1] >= starting_soc * E_BESS_cap, name="BESS_TerminalSOC")

    return {
        "E_BESS_cap": E_BESS_cap,
        "P_BESS_ch_inst": P_BESS_ch_inst,
        "P_BESS_dis_inst": P_BESS_dis_inst,
        "E_BESS": E_BESS,
        "f_BESS": f_BESS,
        "p_BESS": p_BESS,
        "x_BESS": x_BESS,
    }