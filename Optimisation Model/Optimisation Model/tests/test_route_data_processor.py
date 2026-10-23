"""Simple verification tests V11-V19 for the route data processor."""

import pandas as pd
import pytest

from route_data_processor import RouteDataProcessor


def make_processor(tmp_path, rows, days=1):
    """Save a small route schedule and run the processor."""
    csv_file = tmp_path / "route_schedule.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    return RouteDataProcessor(csv_file, days, range(days * 24))


def test_v11_route_times(tmp_path):
    rows = [
        {"Truck No": 1, "Pick Up Time": "00:00", "Drop Off Time": "01:00", "Diesel Used": 10},
        {"Truck No": 2, "Pick Up Time": "07:30", "Drop Off Time": "09:30", "Diesel Used": 10},
        {"Truck No": 3, "Pick Up Time": "08:59", "Drop Off Time": "23:59", "Diesel Used": 10},
    ]
    p = make_processor(tmp_path, rows)
    assert p.time_to_hours("07:30") == 7.5
    assert p.time_to_hours("08:59") == 8 + 59 / 60
    assert p.routes_df["departure_hour"].tolist() == [0, 7, 8]
    assert p.routes_df["return_hour"].tolist() == [1, 10, 24]


def test_v12_repeat_routes(tmp_path):
    rows = [
        {"Truck No": 1, "Pick Up Time": "08:00", "Drop Off Time": "10:00", "Diesel Used": 10},
        {"Truck No": 2, "Pick Up Time": "11:00", "Drop Off Time": "12:00", "Diesel Used": 20},
    ]
    p = make_processor(tmp_path, rows, days=2)
    assert p.routes == [1, 2, 3, 4]
    assert p.routes_df["day"].tolist() == [0, 0, 1, 1]
    assert p.route_year == {1: 0, 2: 0, 3: 0, 4: 0}
    assert p.route_departure == {1: 8, 2: 11, 3: 32, 4: 35}
    assert p.route_return == {1: 10, 2: 12, 3: 34, 4: 36}

    # Check the year changes after day 364.
    p_year = make_processor(tmp_path, rows[:1], days=366)
    assert p_year.route_year[365] == 0
    assert p_year.route_year[366] == 1
    assert p_year.route_departure[366] == 365 * 24 + 8


def test_v13_truck_assignments(tmp_path):
    rows = [
        {"Truck No": 3, "Pick Up Time": "08:00", "Drop Off Time": "09:00", "Diesel Used": 10},
        {"Truck No": 1, "Pick Up Time": "10:00", "Drop Off Time": "11:00", "Diesel Used": 20},
    ]
    p = make_processor(tmp_path, rows, days=2)
    assert p.trucks == [1, 3]
    assert p.route_truck == {1: 3, 2: 1, 3: 3, 4: 1}


def test_v14_diesel_used(tmp_path):
    rows = [
        {"Truck No": 1, "Pick Up Time": "08:00", "Drop Off Time": "09:00", "Diesel Used": 0},
        {"Truck No": 2, "Pick Up Time": "10:00", "Drop Off Time": "11:00", "Diesel Used": 12.5},
    ]
    p = make_processor(tmp_path, rows, days=2)
    assert p.route_diesel_used == {1: 0.0, 2: 12.5, 3: 0.0, 4: 12.5}


def test_v15_route_energy(tmp_path):
    rows = [
        {"Truck No": i + 1, "Pick Up Time": "08:00", "Drop Off Time": "09:00", "Diesel Used": litres}
        for i, litres in enumerate([0, 20, 50])
    ]
    p = make_processor(tmp_path, rows)
    for route, litres in zip(p.routes, [0, 20, 50]):
        assert p.route_energy[route] == pytest.approx(litres * 4.469729345, abs=1e-6)


def test_v16_active_routes_and_trucks(tmp_path):
    rows = [
        {"Truck No": 2, "Pick Up Time": "08:00", "Drop Off Time": "10:00", "Diesel Used": 10},
        {"Truck No": 1, "Pick Up Time": "09:00", "Drop Off Time": "11:00", "Diesel Used": 10},
    ]
    p = make_processor(tmp_path, rows)
    expected_routes = {8: [1], 9: [1, 2], 10: [2]}
    expected_trucks = {8: [2], 9: [1, 2], 10: [1]}
    for t in p.times:
        assert p.active_routes_at[t] == expected_routes.get(t, [])
        assert p.active_trucks_at[t] == expected_trucks.get(t, [])


def test_v17_overlapping_routes(tmp_path):
    rows = [
        {"Truck No": 1, "Pick Up Time": "08:00", "Drop Off Time": "10:00", "Diesel Used": 10},
        {"Truck No": 2, "Pick Up Time": "09:00", "Drop Off Time": "11:00", "Diesel Used": 10},
        {"Truck No": 3, "Pick Up Time": "11:00", "Drop Off Time": "12:00", "Diesel Used": 10},
    ]
    p = make_processor(tmp_path, rows, days=2)
    assert p.overlapping_routes == [(1, 2), (4, 5)]


def test_v18_departures_and_pre_departure(tmp_path):
    rows = [
        {"Truck No": 1, "Pick Up Time": "00:00", "Drop Off Time": "01:00", "Diesel Used": 10},
        {"Truck No": 2, "Pick Up Time": "08:00", "Drop Off Time": "10:00", "Diesel Used": 10},
    ]
    p = make_processor(tmp_path, rows)
    assert p.routes_departing_at[0] == [1]
    assert p.routes_departing_at[8] == [2]
    assert all(p.routes_departing_at[t] == [] for t in p.times if t not in (0, 8))
    assert p.route_pre_departure == {1: 0, 2: 7}


def test_v19_invalid_inputs(tmp_path):
    valid = {"Truck No": 1, "Pick Up Time": "08:00", "Drop Off Time": "10:00", "Diesel Used": 10}

    for column in valid:
        bad_row = {key: value for key, value in valid.items() if key != column}
        with pytest.raises(KeyError):
            make_processor(tmp_path, [bad_row])

    with pytest.raises((ValueError, TypeError)):
        make_processor(tmp_path, [{**valid, "Pick Up Time": "not a time"}])

    with pytest.raises((ValueError, TypeError)):
        make_processor(tmp_path, [{**valid, "Diesel Used": "not a number"}])