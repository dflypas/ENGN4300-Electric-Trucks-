"""
battery_storage_block.py

Stationary BESS part - optimiser.py calls this instead of building BESS
variables/constraints inline.

Matches Section 8 (Stationary BESS, B1-B8) of the constraints document.
Every constraint below is built once as an array (using addMVar), not
once per hour in a loop.
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

    # times must be in order (t, t+1, t+2, ...) for the SOC balance below
    time_positions = [prev_t[t] for t in times[1:]]
    assert time_positions == list(times[:-1]), "times must be contiguous and ordered"

    # pv_generated_t is a {t: value} dict - convert to a plain array in
    # time order so it lines up with the MVar arrays below
    pv_array = np.array([pv_generated_t[t] for t in times])

    if max_charge_power_kw is None:
        max_charge_power_kw = charge_c_rate * max_capacity_kwh
    if max_discharge_power_kw is None:
        max_discharge_power_kw = discharge_c_rate * max_capacity_kwh

    # B1: installed energy capacity (kWh)
    E_BESS_cap = model.addVar(lb=0, ub=max_capacity_kwh, name="Capacity_BESS_kWh")

    # B2: installed charge/discharge power capacity (kW)
    P_BESS_ch_inst = model.addVar(lb=0, ub=max_charge_power_kw, name="ChargePower_Installed_kW")
    P_BESS_dis_inst = model.addVar(lb=0, ub=max_discharge_power_kw, name="DischargePower_Installed_kW")

    # energy stored each hour, and charge/discharge power each hour
    E_BESS = model.addMVar(n_t, lb=0, name="Energy_BESS_kWh")
    f_BESS = model.addMVar(n_t, lb=0, name="BESS_ChargeIn_kWh")
    p_BESS = model.addMVar(n_t, lb=0, name="BESS_DischargeOut_kWh")

    # 1 = charging this hour, 0 = discharging this hour
    x_BESS = model.addMVar(n_t, vtype=GRB.BINARY, name="BESS_ChargeMode")

    # B3: SOC stays within min/max of installed capacity
    model.addConstr(E_BESS >= soc_min * E_BESS_cap, name="BESS_SOCMin")
    model.addConstr(E_BESS <= soc_max * E_BESS_cap, name="BESS_SOCMax")

    # B4: charging limits - installed power, fixed mode gate, C-rate
    model.addConstr(f_BESS <= P_BESS_ch_inst * dt, name="BESS_ChargeInstalledLimit")
    model.addConstr(f_BESS <= max_charge_power_kw * dt * x_BESS, name="BESS_ChargeMode")
    model.addConstr(f_BESS <= charge_c_rate * E_BESS_cap * dt, name="BESS_ChargeRate")

    # B5: discharging limits, same pattern as B4
    model.addConstr(p_BESS <= P_BESS_dis_inst * dt, name="BESS_DischargeInstalledLimit")
    model.addConstr(p_BESS <= max_discharge_power_kw * dt * (1 - x_BESS), name="BESS_DischargeMode")
    model.addConstr(p_BESS <= discharge_c_rate * E_BESS_cap * dt, name="BESS_DischargeRate")

    # BESS charge/discharge power must come from/go to the site's energy flows
    model.addConstr(
        f_BESS - p_BESS == grid_import + pv_array - pv_export - sb_charging_energy,
        name="BESS_SiteLink"
    )

    # B7: SOC balance with efficiency losses - first hour, then every hour after
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

    # B8: can't end the horizon with less energy than the BESS started with
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