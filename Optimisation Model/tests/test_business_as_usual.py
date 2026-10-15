#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:25:19 2026

@author: mollyrobertson
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from business_as_usual import business_as_usual, replacement_annualisation


def get_test_inputs():
    return {"routes": [0, 1, 2],
        "route_diesel_used": {0: 100, 1: 150,2: 200},
        "route_year": {0: 0, 1: 0,2: 0},
        "diesel_price_t": {0: 2.0},
        "trucks": [0, 1],
        "years": [0],
        "trucks_in_year": {0: [0, 1]},
        "diesel_capex": 100000,
        "diesel_lifetime": 5,
        "diesel_om": 0.05,
        "discount_rate": 0.05,
        "project_lifetime": 10}

def test_diesel_consumption():
    inputs = get_test_inputs()
    results, df_results = business_as_usual(**inputs)

    # Expected diesel consumption
    expected_diesel = 100 + 150 + 200
    assert results["diesel_used_L"] == expected_diesel
    
    
def test_diesel_fuel_cost():
    inputs = get_test_inputs()
    results, df_results = business_as_usual(**inputs)
    expected_cost = (100 * 2.0 + 150 * 2.0 + 200 * 2.0)

    assert results["diesel_fuel_cost"] == expected_cost
    assert results["operation_cost"] == expected_cost
    
    
def test_annualised_diesel_capex():
    inputs = get_test_inputs()
    results, df_results = business_as_usual(**inputs)

    r = inputs["discount_rate"]
    n = inputs["project_lifetime"]
    crf = (r * (1 + r)**n / ((1 + r)**n - 1))
    expected_capex = (crf * inputs["diesel_capex"] * len(inputs["trucks"]))

    assert np.isclose(results["annualised_capex"],expected_capex)
    
    
def test_diesel_replacement_cost():
    inputs = get_test_inputs()
    results, df_results = business_as_usual(**inputs)
    expected_replacement = replacement_annualisation(unit_cost=inputs["diesel_capex"], project_lifetime=inputs["project_lifetime"], discount_rate=inputs["discount_rate"], technology_lifetime=inputs["diesel_lifetime"])
    expected_total = (expected_replacement * len(inputs["trucks"]))

    assert np.isclose(results["replacement_cost"],expected_total)
    
def test_diesel_om_cost():
    inputs = get_test_inputs()
    results, df_results = business_as_usual(**inputs)
    expected_om = (inputs["diesel_capex"]* inputs["diesel_om"]* len(inputs["trucks"]))
    assert results["OM_cost"] == expected_om
    
def test_total_cost():
    inputs = get_test_inputs()
    results, df_results = business_as_usual(**inputs)
    expected_total = (results["operation_cost"]+ results["annualised_capex"] + results["replacement_cost"] + results["OM_cost"])
    assert np.isclose(results["total_cost"],expected_total)
    
    
def test_results_dataframe():
    inputs = get_test_inputs()
    results, df_results = business_as_usual(**inputs)

    assert isinstance(results, dict)
    assert isinstance(df_results, pd.DataFrame)
    assert len(df_results) == 1
    assert "total_cost" in df_results.columns
    assert "diesel_used_L" in df_results.columns
    assert "number_of_diesel_trucks" in df_results.columns
    assert np.isclose(df_results.loc[0, "total_cost"], results["total_cost"])
    
    
def test_diesel_price_changes_between_years():
    inputs = get_test_inputs()
    inputs["route_year"] = {0: 0, 1: 0, 2: 1}
    inputs["diesel_price_t"] = {0: 2.0, 1: 2.5}

    results, _ = business_as_usual(**inputs)
    expected_cost = (100 * 2.0+ 150 * 2.0+ 200 * 2.5)
    assert results["diesel_fuel_cost"] == expected_cost

def test_no_replacement_required():
    replacement_cost = replacement_annualisation(unit_cost=100000, project_lifetime=10, discount_rate=0.05, technology_lifetime=15)
    assert replacement_cost == 0


    