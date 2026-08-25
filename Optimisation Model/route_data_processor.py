#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DairyDrive Electric
Route Data Processor

Reads the raw milk collection route schedule and calculates:
    - rounded route departure and return times
    - route duration
    - route electricity requirement
    - route activity
    - route departure and ending information

The processed route data is returned to the optimisation model.
"""

import pandas as pd
import math


# ============================================================
# Input Files
# ============================================================

INPUT_CSV = "input_data.csv"


# ============================================================
# Helper Function
# ============================================================

def time_to_hours(time_value):
    time_value = str(time_value).strip()
    hours, minutes = time_value.split(":")

    return int(hours) + int(minutes) / 60


# ============================================================
# Main Route Processing Function
# ============================================================

def process_route_data(route_csv):

    # ========================================================
    # 1. Read Model Input Parameters
    # ========================================================

    inputs_df = pd.read_csv(INPUT_CSV)

    # Convert the field_name and value columns into a dictionary
    inputs = inputs_df.set_index("field_name")["value"].to_dict()

    # Duration of optimisation study
    # Units: days
    DURATION_DAYS = int(float(inputs["duration"]))
    
    # Empty truck electricity consumption
    # Units: kWh/hour
    P_EMPTY = float(inputs["empty_truck_consumption"])
    
    # Additional electricity consumption for every 1000 L carried
    # Units: kWh/hour per 1000 L
    P_PAYLOAD_1000L = float(inputs["1000L_truck_consumption"])


    # ========================================================
    # 2. Read Raw Route Data
    # ========================================================

    routes_df = pd.read_csv(route_csv)

    # Rename required columns into names used internally
    routes_df = routes_df.rename(columns={
        "Route No": "route_no",
        "Vol": "volume_l",
        "Pick Up Time": "departure_time",
        "Drop Off Time": "return_time"
    })


    # ========================================================
    # 3. Check Required Columns Exist
    # ========================================================

    required_columns = [
        "route_no",
        "volume_l",
        "departure_time",
        "return_time"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in routes_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required route columns: {missing_columns}"
        )


    # ========================================================
    # 4. Convert Route Times to Decimal Hours
    # ========================================================

    routes_df["departure_decimal"] = (
        routes_df["departure_time"]
        .apply(time_to_hours)
    )

    routes_df["return_decimal"] = (
        routes_df["return_time"]
        .apply(time_to_hours)
    )


    # ========================================================
    # 5. Round Route Times to Hourly Timesteps
    # ========================================================

    # Departure is rounded DOWN to the previous whole hour.
    #
    # Example:
    # 05:40 -> 05:00

    routes_df["departure_hour"] = (
        routes_df["departure_decimal"]
        .apply(math.floor)
        .astype(int)
    )


    # Return is rounded UP to the following whole hour.
    #
    # Example:
    # 06:20 -> 07:00

    routes_df["return_hour"] = (
        routes_df["return_decimal"]
        .apply(math.ceil)
        .astype(int)
    )


    # ========================================================
    # 6. Calculate Rounded Route Duration
    # ========================================================

    # T_r = T_return,r - T_departure,r

    routes_df["duration_hr"] = (
        routes_df["return_hour"]
        - routes_df["departure_hour"]
    )


    # ========================================================
    # 7. Calculate Route Energy Requirement
    # ========================================================

    # --------------------------------------------------------
    # Empty half of route
    #
    # E_r,empty = (T_r / 2) * P_empty
    # --------------------------------------------------------

    routes_df["energy_empty_kwh"] = (
        routes_df["duration_hr"]
        / 2
        * P_EMPTY
    )


    # --------------------------------------------------------
    # Loaded half of route
    #
    # E_r,loaded =
    #
    # (T_r / 2)
    # *
    # (P_empty + P_payload * V_r / 1000)
    # --------------------------------------------------------

    routes_df["energy_loaded_kwh"] = (
        routes_df["duration_hr"]
        / 2
        * (
            P_EMPTY
            + P_PAYLOAD_1000L
            * routes_df["volume_l"]
            / 1000
        )
    )


    # --------------------------------------------------------
    # Total route electricity requirement
    #
    # E_r = E_r,empty + E_r,loaded
    # --------------------------------------------------------

    routes_df["route_energy_kwh"] = (
        routes_df["energy_empty_kwh"]
        + routes_df["energy_loaded_kwh"]
    )


    # ========================================================
    # 8. Create Hourly Optimisation Time Set
    # ========================================================
    
    number_of_hours = DURATION_DAYS * 24
    
    times = list(range(number_of_hours))
    
    first_t = times[0]
    last_t = times[-1]
    
    prev_t = {
        times[i]: times[i - 1]
        for i in range(1, len(times))
    }
    
    
    # ========================================================
    # 9. Repeat Daily Route Schedule Across Study Duration
    # ========================================================
    
    daily_routes_df = routes_df.copy()
    
    expanded_routes = []
    
    new_route_no = 1
    
    for day in range(DURATION_DAYS):
    
        for _, row in daily_routes_df.iterrows():
    
            route = row.copy()
    
            # Store which day this route occurs on
            route["day"] = day
    
            # Give every repeated route a unique route number
            route["route_no"] = new_route_no
    
            # Keep original route number for reference
            route["original_route_no"] = int(row["route_no"])
    
            # Convert daily route times into absolute
            # optimisation timesteps
            route["departure_t"] = (
                int(row["departure_hour"])
                + day * 24
            )
    
            route["return_t"] = (
                int(row["return_hour"])
                + day * 24
            )
    
            expanded_routes.append(route)
    
            new_route_no += 1
    
    
    # Convert repeated routes back into a dataframe
    routes_df = pd.DataFrame(expanded_routes)


    # ========================================================
    # 10. Create Route Set
    # ========================================================
    
    routes = (
        routes_df["route_no"]
        .astype(int)
        .tolist()
    )

    # ========================================================
    # 11. Route Departure Dictionary
    # ========================================================
    
    route_departure = (
        routes_df
        .set_index("route_no")["departure_t"]
        .astype(int)
        .to_dict()
    )

    # ========================================================
    # 11. Route Energy Dictionary
    # ========================================================

    # Example:
    #
    # route_energy[1] = 420.0
    #
    # means route 1 requires 420 kWh.

    route_energy = (
        routes_df
        .set_index("route_no")["route_energy_kwh"]
        .to_dict()
    )


    # ========================================================
    # 12. Routes Departing at Each Time
    # ========================================================

    # Your optimiser currently uses:
    #
    # routes_departing_at[r] == t
    #
    # Therefore this dictionary maps:
    #
    # route -> departure timestep

    routes_departing_at = route_departure.copy()

    
    # ========================================================
    # 14. Routes Ending at Each Time
    # ========================================================
    
    routes_ending = {
        t: []
        for t in times
    }
    
    for _, row in routes_df.iterrows():
    
        r = int(row["route_no"])
        return_t = int(row["return_t"])
    
        if return_t in routes_ending:
            routes_ending[return_t].append(r)

    # ========================================================
    # 15. Route Activity Dictionary
    # ========================================================
    
    route_active = {}
    
    for _, row in routes_df.iterrows():
    
        r = int(row["route_no"])
    
        departure_t = int(row["departure_t"])
        return_t = int(row["return_t"])
    
        for t in times:
    
            if departure_t <= t < return_t:
                route_active[r, t] = 1
    
            else:
                route_active[r, t] = 0

    # ========================================================
    # 15. Dates
    # ========================================================

    dates = list(range(DURATION_DAYS))


    # ========================================================
    # 16. Candidate Batteries and Chargers
    # ========================================================

    # Maximum possible number of batteries is taken as one
    # battery per route.
    #
    # The optimiser will decide how many of these candidates
    # are actually required.

    batteries = list(
        range(1, len(routes) + 1)
    )


    # Candidate charger set.
    #
    # The optimiser currently selects the actual number of
    # chargers using N_CS.

    chargers = list(
        range(1, len(routes) + 1)
    )


    # ========================================================
    # 17. Display Processed Route Data
    # ========================================================

    print("\nProcessed Route Data")
    print("=" * 100)

    print(
        routes_df[
            [
                "route_no",
                "volume_l",
                "departure_time",
                "return_time",
                "departure_hour",
                "return_hour",
                "duration_hr",
                "energy_empty_kwh",
                "energy_loaded_kwh",
                "route_energy_kwh"
            ]
        ].to_string(index=False)
    )


    # ========================================================
    # 18. Return Data to Optimiser
    # ========================================================

    return (
        routes_df,
        routes,
        dates,
        times,
        first_t,
        last_t,
        prev_t,
        route_departure,
        routes_ending,
        route_energy,
        route_active,
        routes_departing_at,
        batteries,
        chargers
    )