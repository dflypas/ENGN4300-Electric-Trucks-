"""
Simple constraint checks for DairyDrive Electric.

The optimiser is run once for a short 1-day case. The five tests then check
only the first six timesteps (plus the final SOC condition where relevant).

Run with:
    pytest -q test_constraints_simple.py
"""

import os
from unittest.mock import patch

import gurobipy as gp
import pandas as pd
import pytest

import main


TOL = 1e-5
CHECK_TIMES = range(6)


@pytest.fixture(scope="module")
def solved_model():
    """Run one small model and keep the solved Gurobi model for the tests."""

    captured = {}
    original_optimize = gp.Model.optimize

    def capture_model(model, *args, **kwargs):
        captured["model"] = model
        model.setParam("OutputFlag", 0)
        return original_optimize(model, *args, **kwargs)

    overrides = {
        "duration": 1,
        "max_candidate_batteries": 2,
        "max_candidate_chargers": 2,
    }

    with patch.object(gp.Model, "optimize", capture_model):
        results, _ = main.run_pipeline(
            overrides=overrides,
            time_limit=30,
            mip_gap=0.01,
        )

    assert results is not None

    inputs = pd.read_csv(
        os.path.join(main.INPUT_DIR, "input_data_user_naming.csv")
    ).set_index("field_name")["value"].to_dict()

    inputs.update(overrides)

    return captured["model"], inputs


def v(model, name):
    """Return the solved value of a Gurobi variable."""
    variable = model.getVarByName(name)
    assert variable is not None, f"Variable not found: {name}"
    return variable.X


# V52 ------------------------------------------------------------------------
def test_depot_power_balance(solved_model):
    model, _ = solved_model

    for t in CHECK_TIMES:
        supply = (
            v(model, f"P_G_imp[{t}]")
            + v(model, f"P_PV_generated[{t}]")
            + v(model, f"P_BESS_dis[{t}]")
        )

        demand = (
            sum(v(model, f"P_SB_ch[{t},{b}]") for b in range(2))
            + v(model, f"P_BESS_ch[{t}]")
            + v(model, f"P_G_exp[{t}]")
        )

        assert abs(supply - demand) < TOL


# V53 ------------------------------------------------------------------------
def test_swappable_battery_soc(solved_model):
    model, inputs = solved_model

    capacity = float(inputs["SB_cap_max"])
    minimum = float(inputs["SOC_SB_min"]) * capacity
    maximum = float(inputs["SOC_SB_max"]) * capacity
    initial = float(inputs["SOC_SB_0"]) * capacity

    for b in range(2):
        purchased = v(model, f"w[{b}]")

        assert v(model, f"E_SB[0,{b}]") == pytest.approx(
            initial * purchased,
            abs=TOL,
        )

        for t in CHECK_TIMES:
            energy = v(model, f"E_SB[{t},{b}]")
            assert energy >= minimum * purchased - TOL
            assert energy <= maximum * purchased + TOL


# V54 ------------------------------------------------------------------------
def test_charger_utilisation(solved_model):
    model, inputs = solved_model

    chargers = v(model, "N_CS")
    charger_power = float(inputs["C_charge"])
    battery_power_limit = float(inputs["SB_max_charge_power"])

    for t in CHECK_TIMES:
        charging_count = sum(
            v(model, f"x_SB_ch[{t},{b}]") for b in range(2)
        )
        charging_power = sum(
            v(model, f"P_SB_ch[{t},{b}]") for b in range(2)
        )

        assert charging_count <= chargers + TOL
        assert charging_power <= chargers * charger_power + TOL

        for b in range(2):
            assert v(model, f"P_SB_ch[{t},{b}]") <= battery_power_limit + TOL


# V55 ------------------------------------------------------------------------
def test_grid_import_export(solved_model):
    model, inputs = solved_model

    grid_capacity = v(model, "S_G")
    max_import = float(inputs["GRID_max_import"])
    max_export = float(inputs["GRID_max_export"])

    for t in CHECK_TIMES:
        grid_import = v(model, f"P_G_imp[{t}]")
        grid_export = v(model, f"P_G_exp[{t}]")

        assert grid_import <= grid_capacity + TOL
        assert grid_export <= grid_capacity + TOL
        assert grid_import <= max_import + TOL
        assert grid_export <= max_export + TOL

        assert not (
            grid_import > TOL
            and grid_export > TOL
        )


# V56 ------------------------------------------------------------------------
def test_bess_operation(solved_model):
    model, inputs = solved_model

    capacity = v(model, "Q_BESS")
    minimum = float(inputs["BESS_min_soc"]) * capacity
    maximum = float(inputs["BESS_max_soc"]) * capacity
    max_charge = float(inputs["BESS_charge"])
    max_discharge = float(inputs["BESS_discharge"])

    for t in CHECK_TIMES:
        energy = v(model, f"E_BESS[{t}]")
        charge = v(model, f"P_BESS_ch[{t}]")
        discharge = v(model, f"P_BESS_dis[{t}]")

        assert energy >= minimum - TOL
        assert energy <= maximum + TOL
        assert charge <= max_charge + TOL
        assert discharge <= max_discharge + TOL

        assert not (
            charge > TOL
            and discharge > TOL
        )

    initial_soc = float(inputs["BESS_initial_soc"])
    assert v(model, "E_BESS[0]") == pytest.approx(
        initial_soc * capacity,
        abs=TOL,
    )

    assert v(model, "E_BESS[23]") >= v(model, "E_BESS[0]") - TOL
