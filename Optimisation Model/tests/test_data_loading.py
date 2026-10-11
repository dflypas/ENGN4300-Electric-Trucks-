#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:09:47 2026

@author: mollyrobertson
"""


import json
import os
import math
from functools import lru_cache
import pandas as pd
import numpy as np
import matplotlib as plt
import pvlib
from pvlib import pvsystem
import datetime as dt


import pytest

import time 
INPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# V01
# Confirm the user-input CSV is read correctly and required fields can be converted to the expected Python types.

@pytest.mark.parametrize("input_file", [
    "input_data_correct.csv",
    "input_data_string.csv",
    "input_data_integer.csv"
])

def test_v01_valid_input_loading(input_file):

    # ============================================================
    # Load Input Data - same as main.py
    # ============================================================

    input_data = pd.read_csv(os.path.join(INPUT_DIR, input_file))
    inputs = input_data.set_index("field_name")["value"].to_dict()

    # ============================================================
    # Validate Input Data - subset of main.py
    # ============================================================

    int_fields = ["duration","year","n_trucks","max_candidate_batteries","max_candidate_chargers"]
    float_fields = ["dt","latitude", "longitude","SB_cap_max","SOC_SB_min","SOC_SB_max","SOC_SB_0"]

    for field in int_fields + float_fields:
        if field not in inputs:
            raise ValueError(f"Missing required input: '{field}'")

    for field in int_fields:
        try:
            value = float(inputs[field])
            if not value.is_integer():
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(
                f"Input '{field}' must be an integer. Received: {inputs[field]}"
            )

    for field in float_fields:
        try:
            float(inputs[field])
        except (ValueError, TypeError):
            raise ValueError(
                f"Input '{field}' must be numeric. Received: {inputs[field]}"
            )

    # ============================================================
    # Define Static Input Data - subset of main.py
    # ============================================================

    PROJECT_DURATION_DAYS = int(inputs["duration"])
    DT = float(inputs["dt"])
    YEAR = int(inputs["year"])

    LATITUDE = float(inputs["latitude"])
    LONGITUDE = float(inputs["longitude"])

    N_TRUCKS = int(inputs["n_trucks"])

    MAX_CANDIDATE_BATTERIES = int(inputs["max_candidate_batteries"])
    MAX_CANDIDATE_CHARGERS = int(inputs["max_candidate_chargers"])

    SB_CAP_MAX = float(inputs["SB_cap_max"])
    SOC_SB_MIN = float(inputs["SOC_SB_min"])
    SOC_SB_MAX = float(inputs["SOC_SB_max"])
    SOC_SB_0 = float(inputs["SOC_SB_0"])

    # ============================================================
    # Check Expected Results
    # ============================================================

    assert PROJECT_DURATION_DAYS == 100
    assert DT == pytest.approx(1.0)
    assert YEAR == 2026

    assert LATITUDE == pytest.approx(-36.674)
    assert LONGITUDE == pytest.approx(149.842)

    assert N_TRUCKS == 8

    assert MAX_CANDIDATE_BATTERIES == 8
    assert MAX_CANDIDATE_CHARGERS == 8

    assert SB_CAP_MAX == pytest.approx(650)
    assert SOC_SB_MIN == pytest.approx(0.2)
    assert SOC_SB_MAX == pytest.approx(1.0)
    assert SOC_SB_0 == pytest.approx(0.8)

    # Check Python types
    assert isinstance(PROJECT_DURATION_DAYS, int)
    assert isinstance(YEAR, int)
    assert isinstance(N_TRUCKS, int)
    assert isinstance(MAX_CANDIDATE_BATTERIES, int)
    assert isinstance(MAX_CANDIDATE_CHARGERS, int)

    assert isinstance(DT, float)
    assert isinstance(LATITUDE, float)
    assert isinstance(LONGITUDE, float)
    assert isinstance(SB_CAP_MAX, float)
    assert isinstance(SOC_SB_MIN, float)
    assert isinstance(SOC_SB_MAX, float)
    assert isinstance(SOC_SB_0, float)



# V03
# Confirm duration, year and timestep sets are constructed correctly
# from duration and dt.

@pytest.mark.parametrize(
    "duration, dt, expected_years, expected_timesteps",
    [
        # 1. Test 1 day at dt = 1 hour
        (1,   1.0, 1, 8760),

        # 2. Test 365 days at dt = 1 hour
        (365, 1.0, 1, 8760),

        # 3. Test 366 days - should create a second project year
        (366, 1.0, 2, 8760),

        # 4. Test a non-default timestep
        (365, 0.5, 1, 17520),
    ]
)

def test_v03_time_sets(
    duration,
    dt,
    expected_years,
    expected_timesteps
):

    # ============================================================
    # Construct Time Sets - same as main.py
    # ============================================================

    PROJECT_DURATION_DAYS = duration
    N_YEARS = max(1,math.ceil(PROJECT_DURATION_DAYS / 365))
    YEARS = list(range(N_YEARS))
    DT = dt
    DURATION_DAYS = 365
    NUMBER_OF_TIMESTEPS = int(DURATION_DAYS * 24 / DT)
    TIMES = list(range(NUMBER_OF_TIMESTEPS))
    FIRST_T = TIMES[0]
    LAST_T = TIMES[-1]

    # ============================================================
    # Compare Against Hand Calculations
    # ============================================================

    assert N_YEARS == expected_years

    assert YEARS == list(range(expected_years))

    assert NUMBER_OF_TIMESTEPS == expected_timesteps

    assert TIMES == list(range(expected_timesteps))

    assert FIRST_T == 0

    assert LAST_T == expected_timesteps - 1



# V04
# Confirm candidate battery and charger index sets are created
# correctly from the user-specified maxima.

@pytest.mark.parametrize(
    "max_batteries, max_chargers",
    [
        (8, 8),   # Normal case
        (1, 1),   # Boundary case: one candidate
        (0, 0),   # Boundary case: zero candidates
        (5, 3),   # Different battery and charger maxima
    ]
)
def test_v04_candidate_sets(max_batteries, max_chargers):

    # ============================================================
    # Construct Candidate Sets - same as main.py
    # ============================================================

    MAX_CANDIDATE_BATTERIES = max_batteries
    MAX_CANDIDATE_CHARGERS = max_chargers

    BATTERIES = list(range(MAX_CANDIDATE_BATTERIES))
    CHARGERS = list(range(MAX_CANDIDATE_CHARGERS))

    # ============================================================
    # Check Expected Results
    # ============================================================

    # Correct number of candidates
    assert len(BATTERIES) == MAX_CANDIDATE_BATTERIES
    assert len(CHARGERS) == MAX_CANDIDATE_CHARGERS

    # Correct zero-based indices
    assert BATTERIES == list(range(max_batteries))
    assert CHARGERS == list(range(max_chargers))

    # No duplicate indices
    assert len(BATTERIES) == len(set(BATTERIES))
    assert len(CHARGERS) == len(set(CHARGERS))

    # Check first and last indices when candidates exist
    if max_batteries > 0:
        assert BATTERIES[0] == 0
        assert BATTERIES[-1] == max_batteries - 1
    else:
        assert BATTERIES == []

    if max_chargers > 0:
        assert CHARGERS[0] == 0
        assert CHARGERS[-1] == max_chargers - 1
    else:
        assert CHARGERS == []



    