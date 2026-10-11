#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:23:36 2026

@author: mollyrobertson
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from route_data_processor import RouteDataProcessor


def test_v11_route_time_conversion(tmp_path):

    # ============================================================
    # Create route fixture
    # ============================================================

    route_data = pd.DataFrame({
        "Truck No": [1, 2, 3, 4],
        "Pick Up Time": ["00:00", "08:30", "10:59", "23:00"],
        "Drop Off Time": ["01:00", "10:30", "12:01", "23:59"],
        "Diesel Used": [10, 20, 30, 40]
    })

    route_file = tmp_path / "route_schedule.csv"
    route_data.to_csv(route_file, index=False)

    processor = RouteDataProcessor(
        route_csv=route_file,
        duration_days=1,
        times=list(range(24))
    )

    # ============================================================
    # Test time_to_hours directly
    # ============================================================

    assert processor.time_to_hours("00:00") == 0
    assert processor.time_to_hours("08:30") == 8.5
    assert processor.time_to_hours("10:59") == pytest.approx(10.9833333333)
    assert processor.time_to_hours("23:59") == pytest.approx(23.9833333333)

    # ============================================================
    # Test departure = floor and return = ceil
    # ============================================================

    assert processor.routes_df["departure_hour"].tolist() == [0, 8, 10, 23]
    assert processor.routes_df["return_hour"].tolist() == [1, 11, 13, 24]

    assert processor.route_departure == {
        1: 0,
        2: 8,
        3: 10,
        4: 23
    }

    assert processor.route_return == {
        1: 1,
        2: 11,
        3: 13,
        4: 24
    }
    
    
def test_v12_route_expansion(tmp_path):

    # ============================================================
    # Test two routes repeated across two days
    # ============================================================

    route_data = pd.DataFrame({
        "Truck No": [1, 2],
        "Pick Up Time": ["08:00", "12:00"],
        "Drop Off Time": ["10:00", "14:00"],
        "Diesel Used": [20, 30]
    })

    route_file = tmp_path / "route_schedule.csv"
    route_data.to_csv(route_file, index=False)

    processor = RouteDataProcessor(
        route_csv=route_file,
        duration_days=2,
        times=list(range(48))
    )

    assert processor.routes == [1, 2, 3, 4]

    assert processor.routes_df["day"].astype(int).tolist() == [0, 0, 1, 1]

    assert processor.route_departure == {
        1: 8,
        2: 12,
        3: 32,
        4: 36
    }

    assert processor.route_return == {
        1: 10,
        2: 14,
        3: 34,
        4: 38
    }

    # ============================================================
    # Test year boundary at days 364 and 365
    # ============================================================

    processor_year = RouteDataProcessor(
        route_csv=route_file,
        duration_days=366,
        times=list(range(366 * 24))
    )

    day_364 = processor_year.routes_df[
        processor_year.routes_df["day"] == 364
    ]

    day_365 = processor_year.routes_df[
        processor_year.routes_df["day"] == 365
    ]

    assert (day_364["year"] == 0).all()
    assert (day_365["year"] == 1).all()
    


def test_v13_truck_assignments(tmp_path):

    route_data = pd.DataFrame({
        "Truck No": [5, 2, 8],
        "Pick Up Time": ["08:00", "10:00", "12:00"],
        "Drop Off Time": ["09:00", "11:00", "13:00"],
        "Diesel Used": [10, 20, 30]
    })

    route_file = tmp_path / "route_schedule.csv"
    route_data.to_csv(route_file, index=False)

    processor = RouteDataProcessor(
        route_csv=route_file,
        duration_days=2,
        times=list(range(48))
    )

    # Sorted unique trucks
    assert processor.trucks == [2, 5, 8]

    # Route-to-truck assignment preserved across days
    assert processor.route_truck == {
        1: 5,
        2: 2,
        3: 8,
        4: 5,
        5: 2,
        6: 8
    }
    
def test_v14_route_diesel_used(tmp_path):

    route_data = pd.DataFrame({
        "Truck No": [1, 2, 3],
        "Pick Up Time": ["08:00", "10:00", "12:00"],
        "Drop Off Time": ["09:00", "11:00", "13:00"],
        "Diesel Used": [0.0, 12.5, 27.75]
    })

    route_file = tmp_path / "route_schedule.csv"
    route_data.to_csv(route_file, index=False)

    processor = RouteDataProcessor(
        route_csv=route_file,
        duration_days=2,
        times=list(range(48))
    )

    expected = {
        1: 0.0,
        2: 12.5,
        3: 27.75,
        4: 0.0,
        5: 12.5,
        6: 27.75
    }

    assert processor.route_diesel_used == expected

    for value in processor.route_diesel_used.values():
        assert isinstance(value, float)
        assert value >= 0
        
        
def test_v15_route_energy_calculation(tmp_path):

    route_data = pd.DataFrame({
        "Truck No": [1, 2, 3],
        "Pick Up Time": ["08:00", "10:00", "12:00"],
        "Drop Off Time": ["09:00", "11:00", "13:00"],
        "Diesel Used": [0.0, 20.0, 50.0]
    })

    route_file = tmp_path / "route_schedule.csv"
    route_data.to_csv(route_file, index=False)

    processor = RouteDataProcessor(
        route_csv=route_file,
        duration_days=1,
        times=list(range(24))
    )

    conversion_factor = 4.469729345

    expected = {
        1: 0.0 * conversion_factor,
        2: 20.0 * conversion_factor,
        3: 50.0 * conversion_factor
    }

    for route in processor.routes:
        assert processor.route_energy[route] == pytest.approx(
            expected[route],
            abs=1e-6
        )


def test_v16_active_routes_and_trucks(tmp_path):

    route_data = pd.DataFrame({
        "Truck No": [1, 2],
        "Pick Up Time": ["08:00", "09:00"],
        "Drop Off Time": ["11:00", "12:00"],
        "Diesel Used": [10, 20]
    })

    route_file = tmp_path / "route_schedule.csv"
    route_data.to_csv(route_file, index=False)

    TIMES = list(range(24))

    processor = RouteDataProcessor(
        route_csv=route_file,
        duration_days=1,
        times=TIMES
    )

    expected_routes = {t: [] for t in TIMES}
    expected_routes[8] = [1]
    expected_routes[9] = [1, 2]
    expected_routes[10] = [1, 2]
    expected_routes[11] = [2]

    expected_trucks = {t: [] for t in TIMES}
    expected_trucks[8] = [1]
    expected_trucks[9] = [1, 2]
    expected_trucks[10] = [1, 2]
    expected_trucks[11] = [2]

    assert processor.active_routes_at == expected_routes
    assert processor.active_trucks_at == expected_trucks


def test_v17_overlapping_routes(tmp_path):

    route_data = pd.DataFrame({
        "Truck No": [1, 2, 3, 4],
        "Pick Up Time": ["08:00", "09:00", "12:00", "14:00"],
        "Drop Off Time": ["10:00", "11:00", "14:00", "16:00"],
        "Diesel Used": [10, 20, 30, 40]
    })

    route_file = tmp_path / "route_schedule.csv"
    route_data.to_csv(route_file, index=False)

    processor = RouteDataProcessor(
        route_csv=route_file,
        duration_days=2,
        times=list(range(48))
    )

    # Routes 1 and 2 overlap.
    # Routes 3 and 4 only touch at 14:00, so they do not overlap.
    # Day 2 produces route numbers 5-8.

    expected = [
        (1, 2),
        (5, 6)
    ]

    assert processor.overlapping_routes == expected


def test_v18_departure_mappings(tmp_path):

    route_data = pd.DataFrame({
        "Truck No": [1, 2, 3],
        "Pick Up Time": ["00:00", "05:00", "10:00"],
        "Drop Off Time": ["01:00", "06:00", "11:00"],
        "Diesel Used": [10, 20, 30]
    })

    route_file = tmp_path / "route_schedule.csv"
    route_data.to_csv(route_file, index=False)

    TIMES = list(range(24))

    processor = RouteDataProcessor(
        route_csv=route_file,
        duration_days=1,
        times=TIMES
    )

    expected_departing = {t: [] for t in TIMES}
    expected_departing[0] = [1]
    expected_departing[5] = [2]
    expected_departing[10] = [3]

    assert processor.routes_departing_at == expected_departing

    assert processor.route_pre_departure == {
        1: 0,
        2: 4,
        3: 9
    }


def test_v19_invalid_route_inputs(tmp_path):

    TIMES = list(range(24))

    # ============================================================
    # Missing required column
    # ============================================================

    missing_column = pd.DataFrame({
        "Truck No": [1],
        "Pick Up Time": ["08:00"],
        "Drop Off Time": ["10:00"]
        # Diesel Used missing
    })

    route_file = tmp_path / "missing_column.csv"
    missing_column.to_csv(route_file, index=False)

    with pytest.raises(
        ValueError,
        match="Missing required route column"
    ):
        RouteDataProcessor(route_file, 1, TIMES)

    # ============================================================
    # Invalid time
    # ============================================================

    invalid_time = pd.DataFrame({
        "Truck No": [1],
        "Pick Up Time": ["25:30"],
        "Drop Off Time": ["10:00"],
        "Diesel Used": [20.0]
    })

    route_file = tmp_path / "invalid_time.csv"
    invalid_time.to_csv(route_file, index=False)

    with pytest.raises(
        ValueError,
        match="Invalid time"
    ):
        RouteDataProcessor(route_file, 1, TIMES)

    # ============================================================
    # Non-numeric diesel use
    # ============================================================

    invalid_diesel = pd.DataFrame({
        "Truck No": [1],
        "Pick Up Time": ["08:00"],
        "Drop Off Time": ["10:00"],
        "Diesel Used": ["hello"]
    })

    route_file = tmp_path / "invalid_diesel.csv"
    invalid_diesel.to_csv(route_file, index=False)

    with pytest.raises(
        ValueError,
        match="Diesel Used must contain numeric values"
    ):
        RouteDataProcessor(route_file, 1, TIMES)

    # ============================================================
    # Negative diesel use
    # ============================================================

    negative_diesel = pd.DataFrame({
        "Truck No": [1],
        "Pick Up Time": ["08:00"],
        "Drop Off Time": ["10:00"],
        "Diesel Used": [-20.0]
    })

    route_file = tmp_path / "negative_diesel.csv"
    negative_diesel.to_csv(route_file, index=False)

    with pytest.raises(
        ValueError,
        match="Diesel Used cannot be negative"
    ):
        RouteDataProcessor(route_file, 1, TIMES)