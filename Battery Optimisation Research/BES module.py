"""
battery_storage_block.py

Adds the BESS part to the shared model.
Uses times/first_t/prev_t like the rest of the model, and treats
BESS capacity as something the optimiser solves for, not a fixed input.

Note: charge_power/discharge_power and grid_import/pv_export both
describe energy moving in and out of the BESS. 
"""

import gurobipy as gp
from gurobipy import GRB


def build_bess_block(model, times, first_t, prev_t,
                      soc_min, soc_max,
                      charge_efficiency, discharge_efficiency,
                      max_charge_rate, max_discharge_rate,
                      grid_import, pv_generated_t, pv_export,
                      sb_charging_energy=None,
                      starting_soc=0.5, max_bess_capacity=None):

    # how big the BESS is (solved for, not fixed)
    cap_kwargs = {"lb": 0, "name": "E_BESS_cap"}
    if max_bess_capacity is not None:
        cap_kwargs["ub"] = max_bess_capacity
    E_BESS_cap = model.addVar(**cap_kwargs)

    # energy stored, charge power, discharge power per timestep
    E_BESS = model.addVars(times, lb=0, name="E_BESS")
    charge_power = model.addVars(times, lb=0, ub=max_charge_rate, name="BESS_charge_power")
    discharge_power = model.addVars(times, lb=0, ub=max_discharge_rate, name="BESS_discharge_power")

    # 1 = charging this hour, 0 = discharging this hour
    is_charging = model.addVars(times, vtype=GRB.BINARY, name="BESS_is_charging")

    # keep charge level between soc_min and soc_max
    for t in times:
        model.addConstr(E_BESS[t] >= soc_min * E_BESS_cap, name=f"BESS_soc_min_{t}")
        model.addConstr(E_BESS[t] <= soc_max * E_BESS_cap, name=f"BESS_soc_max_{t}")

    # only charge or discharge, not both, each hour
    for t in times:
        model.addConstr(charge_power[t] <= max_charge_rate * is_charging[t],
                         name=f"BESS_charge_only_if_mode_{t}")
        model.addConstr(discharge_power[t] <= max_discharge_rate * (1 - is_charging[t]),
                         name=f"BESS_discharge_only_if_mode_{t}")

    # energy balance each hour
    for t in times:
        previous_energy = starting_soc * E_BESS_cap if t == first_t else E_BESS[prev_t[t]]

        energy_in = charge_power[t] * charge_efficiency
        energy_out = discharge_power[t] / discharge_efficiency

        sb_draw = sb_charging_energy[t] if sb_charging_energy is not None else 0

        model.addConstr(
            E_BESS[t] == previous_energy
            + grid_import[t]
            + pv_generated_t[t]
            - pv_export[t]
            - sb_draw
            + energy_in
            - energy_out,
            name=f"BESS_energy_balance_{t}"
        )

    # end at the same charge level it started at
    model.addConstr(
        E_BESS[times[-1]] == starting_soc * E_BESS_cap,
        name="BESS_soc_end_closure"
    )

    return {
        "E_BESS_cap": E_BESS_cap,
        "E_BESS": E_BESS,
        "charge_power": charge_power,
        "discharge_power": discharge_power,
        "is_charging": is_charging,
    }
