"""
battery_storage_block.py

Builds the Battery Storage System (BESS) part of the optimisation model.
  - one day, 24 hours (t = 0 to 23)
  - one battery that can either charge OR discharge each hour, not both
  - the battery has a minimum and maximum charge level it must stay between
  - the battery ends the day at the same charge level it started at
  - charging costs money (buying from the grid)
  - discharging saves money (uses stored energy instead of buying)
 
"""

import gurobipy as gp
from gurobipy import GRB


def build_bess_block(model, battery_capacity, soc_min, soc_max,
                      charge_efficiency, discharge_efficiency,
                      max_charge_rate, max_discharge_rate,
                      price_import, hours=24, starting_soc=0.5):
    """
    Adds the battery storage variables and rules to the shared model.

    Inputs:
        model                -- the shared Gurobi model (built elsewhere)
        battery_capacity      -- kWh, size of the battery
        soc_min, soc_max       -- fraction (0 to 1), lowest/highest allowed charge
        charge_efficiency      -- fraction (0 to 1), energy lost while charging
        discharge_efficiency   -- fraction (0 to 1), energy lost while discharging
        max_charge_rate         -- kW, fastest the battery can charge
        max_discharge_rate      -- kW, fastest the battery can discharge
        price_import           -- list of 24 numbers, AUD/kWh to buy power each hour
                                   (discharging is valued at this same price, i.e.
                                   it is treated as avoided grid purchase, not as a
                                   separate grid sale -- there is no export price here)
        hours                  -- number of timesteps in one day (default 24)
        starting_soc            -- fraction (0 to 1), charge level at the start
                                   AND end of the day (default 0.5 = half full)

    Returns:
        a dictionary holding the new variables and the total cost expression,
        so the optimiser wrapper can add this cost into TotalCost.
    """

    # -----------------------------------------------------------------
    # Create the decision variables
    # -----------------------------------------------------------------

    # how much energy is stored in the battery at the END of each hour
    stored_energy = model.addVars(hours, lb=0, ub=battery_capacity, name="stored_energy")

    # how much the battery charges (draws power in) during each hour
    charge_power = model.addVars(hours, lb=0, ub=max_charge_rate, name="charge_power")

    # how much the battery discharges (sends power out) during each hour
    discharge_power = model.addVars(hours, lb=0, ub=max_discharge_rate, name="discharge_power")

    # 1 means "charging this hour", 0 means "discharging this hour"
    # this stops the battery from doing both at the same time
    is_charging = model.addVars(hours, vtype=GRB.BINARY, name="is_charging")

    # -----------------------------------------------------------------
    # Keep the battery within its allowed charge range
    # -----------------------------------------------------------------

    min_allowed = soc_min * battery_capacity
    max_allowed = soc_max * battery_capacity

    for t in range(hours):
        model.addConstr(stored_energy[t] >= min_allowed, name=f"soc_min_{t}")
        model.addConstr(stored_energy[t] <= max_allowed, name=f"soc_max_{t}")

    # -----------------------------------------------------------------
    # Only allow charging OR discharging, not both, each hour
    # -----------------------------------------------------------------

    for t in range(hours):
        # if is_charging[t] = 1, discharge must be 0
        # if is_charging[t] = 0, charge must be 0
        model.addConstr(charge_power[t] <= max_charge_rate * is_charging[t],
                         name=f"charge_only_if_mode_{t}")
        model.addConstr(discharge_power[t] <= max_discharge_rate * (1 - is_charging[t]),
                         name=f"discharge_only_if_mode_{t}")

    # -----------------------------------------------------------------
    # Energy balance: how the stored energy changes hour to hour
    # -----------------------------------------------------------------
    # stored energy this hour = stored energy last hour
    #                          + energy that went IN (after charging losses)
    #                          - energy that went OUT (before discharging losses)

    starting_energy = starting_soc * battery_capacity

    for t in range(hours):
        energy_in = charge_power[t] * charge_efficiency
        energy_out = discharge_power[t] / discharge_efficiency

        if t == 0:
            previous_energy = starting_energy
        else:
            previous_energy = stored_energy[t - 1]

        model.addConstr(
            stored_energy[t] == previous_energy + energy_in - energy_out,
            name=f"energy_balance_{t}"
        )

    # end the day at the same charge level it started at, so this single
    # day can stand in for a longer period without gaining free energy
    model.addConstr(
        stored_energy[hours - 1] == starting_energy,
        name="soc_end_of_day_closure"
    )

    # -----------------------------------------------------------------
    # Work out the cost of running the battery this day
    # -----------------------------------------------------------------
    # charging costs money (buying grid power)
    # discharging saves money (instead of buying grid power at that price)

    cost_of_charging = gp.quicksum(charge_power[t] * price_import[t] for t in range(hours))
    savings_from_discharging = gp.quicksum(discharge_power[t] * price_import[t] for t in range(hours))

    daily_battery_cost = cost_of_charging - savings_from_discharging

    # -----------------------------------------------------------------
    # Return everything the optimiser needs
    # -----------------------------------------------------------------

    return {
        "stored_energy": stored_energy,
        "charge_power": charge_power,
        "discharge_power": discharge_power,
        "is_charging": is_charging,
        "daily_battery_cost": daily_battery_cost,
    }