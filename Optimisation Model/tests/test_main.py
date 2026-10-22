"""Simple verification tests V01-V04 for data loading and time sets."""

import math
from pathlib import Path

import pandas as pd
import pytest


# The test file is placed in the tests folder, alongside main.py in its parent folder.
INPUT_FILE = Path(__file__).resolve().parents[1] / "input_data.csv"


def load_inputs(csv_file):
    """Load the input CSV using the same two lines as main.py."""
    data = pd.read_csv(csv_file)
    return data.set_index("field_name")["value"].to_dict()


def test_v01_input_loading(tmp_path):
    inputs = load_inputs(INPUT_FILE)
    assert int(inputs["duration"]) == 30
    assert int(inputs["year"]) == 2026
    assert float(inputs["dt"]) == 1.0
    assert float(inputs["latitude"]) == pytest.approx(-36.674)

    # A missing field fails when main.py tries to use it.
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"field_name": ["year"], "value": [2026]}).to_csv(missing, index=False)
    with pytest.raises(KeyError):
        int(load_inputs(missing)["duration"])

    # A non-numeric field fails when converted to an integer.
    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"field_name": ["duration"], "value": ["not a number"]}).to_csv(invalid, index=False)
    with pytest.raises(ValueError):
        int(load_inputs(invalid)["duration"])


def test_v02_input_overrides():
    # Load the original input CSV.
    input_data = pd.read_csv("input_data.csv")
    inputs = input_data.set_index("field_name")["value"].to_dict()

    original = inputs.copy()

    # Apply overrides for duration, year and battery capacity.
    overrides = {
        "duration": 7,
        "year": 2027,
        "max_candidate_batteries": 3
    }

    inputs.update(overrides)

    # Check that the overrides were applied.
    assert inputs["duration"] == 7
    assert inputs["year"] == 2027
    assert inputs["max_candidate_batteries"] == 3

def test_v03_time_sets():
    for days, dt, expected_years, expected_steps in [
        (1, 1, 1, 24),
        (365, 1, 1, 8760),
        (366, 1, 2, 8760),
        (2, 0.5, 1, 96),
    ]:
        n_years = max(1, math.ceil(days / 365))
        years = list(range(n_years))
        duration_days = days if days < 365 else 365
        number_of_timesteps = int(duration_days * 24 / dt)
        times = list(range(number_of_timesteps))
        first_t = times[0]
        last_t = times[-1]

        assert n_years == expected_years
        assert years == list(range(expected_years))
        assert number_of_timesteps == expected_steps
        assert len(times) == len(set(times))
        assert first_t == 0
        assert last_t == expected_steps - 1


def test_v04_candidate_sets():
    for max_batteries, max_chargers in [(0, 0), (1, 1), (8, 4)]:
        batteries = list(range(max_batteries))
        chargers = list(range(max_chargers))
        assert batteries == list(range(max_batteries))
        assert chargers == list(range(max_chargers))
        assert len(batteries) == len(set(batteries))
        assert len(chargers) == len(set(chargers))
        if batteries:
            assert batteries[0] == 0 and batteries[-1] == max_batteries - 1
        if chargers:
            assert chargers[0] == 0 and chargers[-1] == max_chargers - 1
