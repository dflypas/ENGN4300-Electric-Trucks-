#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:57:48 2026

@author: mollyrobertson

Route and Time Processor for DairyDrive Electric

This module:
1. Imports the base daily route schedule.
2. Repeats the schedule for 10 consecutive days.
3. Calculates route duration from the true pickup and drop-off times.
4. Calculates route energy assuming constant truck demand of 150 kW.
5. Creates the hourly optimisation timestep set.
6. Creates route departure, return, energy and activity dictionaries.
7. Returns all outputs in the same structure expected by the optimiser.
"""

import pandas as pd

# ============================================================
# Input Values
# ============================================================

TRUCK_ENERGY_USE_KW = 150.0
NUMBER_OF_DAYS = 4
REFERENCE_DATE = pd.Timestamp("2010-01-01")


# ============================================================
# Main Processing Function
# ============================================================

def process_route_data(ROUTE_CSV):

    daily_routes_df = pd.read_csv(ROUTE_CSV)
    daily_routes_df.columns = daily_routes_df.columns.str.strip()

    daily_routes_df["Pick Up Time"] = pd.to_datetime(daily_routes_df["Pick Up Time"])
    daily_routes_df["Drop Off Time"] = pd.to_datetime(daily_routes_df["Drop Off Time"])

    # ============================================================
    # Repeat Route Schedule Daily
    # ============================================================

    routes = []
    route_counter = 1

    for day in range(NUMBER_OF_DAYS):

        current_date = (REFERENCE_DATE + pd.Timedelta(days=day))


        for _, row in daily_routes_df.iterrows():

            # ====================================================
            # Original exact route times
            # ====================================================

            exact_departure = (
                current_date
                + pd.Timedelta(
                    hours=row["Pick Up Time"].hour,
                    minutes=row["Pick Up Time"].minute
                )
            )

            exact_return = (
                current_date
                + pd.Timedelta(
                    hours=row["Drop Off Time"].hour,
                    minutes=row["Drop Off Time"].minute
                )
            )


            # If route crosses midnight
            if exact_return <= exact_departure:

                exact_return += pd.Timedelta(days=1)


            # ====================================================
            # Convert to Hourly Optimisation Times
            # ====================================================

            # Departure is rounded DOWN.
            #
            # Example:
            # 05:40 --> 05:00
            #
            # This ensures the battery is considered unavailable
            # for the entire hour containing the departure.

            departure = exact_departure.floor("h")


            # Return is rounded UP.
            #
            # Example:
            # 08:25 --> 09:00
            #
            # This ensures the battery is not considered available
            # before the real route has finished.

            return_time = exact_return.ceil("h")


            # ====================================================
            # Rounded Route Duration
            # ====================================================

            route_duration_hours = (
                return_time - departure
            ).total_seconds() / 3600


            # ====================================================
            # Route Energy Requirement
            # ====================================================

            # Constant truck demand of 150 kW.
            #
            # Because duration has been conservatively rounded,
            # energy demand is also conservatively rounded upward.

            route_energy_kwh = (
                TRUCK_ENERGY_USE_KW
                * route_duration_hours
            )


            # ====================================================
            # Store Processed Route
            # ====================================================

            routes.append({

                "route_no": route_counter,

                "original_route_no": row["Route No"],

                "date": current_date,

                "departure_time": departure,

                "return_time": return_time,

                "route_duration_hours": route_duration_hours,

                "route_energy_kwh": route_energy_kwh,

                "required_energy_kwh": route_energy_kwh
            })


            route_counter += 1


    # ============================================================
    # Processed Route DataFrame
    # ============================================================

    routes_df = pd.DataFrame(routes)


    # ============================================================
    # Route Set
    # ============================================================

    routes = list(routes_df["route_no"])


    # ============================================================
    # Date Set
    # ============================================================

    dates = [
        REFERENCE_DATE + pd.Timedelta(days=day)
        for day in range(NUMBER_OF_DAYS)
    ]


    # ============================================================
    # Hourly Time Set
    # ============================================================

    times = []

    for d in dates:

        for h in range(24):

            times.append(
                pd.Timestamp(d)
                + pd.Timedelta(hours=h)
            )

    times = sorted(times)

    first_t = times[0]
    last_t = times[-1]


    # ============================================================
    # Previous Timestep
    # ============================================================

    prev_t = {
        times[i]: times[i - 1]
        for i in range(1, len(times))
    }


    # ============================================================
    # Route Departure
    # ============================================================

    route_departure = dict(
        zip(
            routes_df["route_no"],
            routes_df["departure_time"]
        )
    )


    # ============================================================
    # Route Return
    # ============================================================

    routes_ending = dict(
        zip(
            routes_df["route_no"],
            routes_df["return_time"]
        )
    )


    # ============================================================
    # Route Energy
    # ============================================================

    route_energy = dict(
        zip(
            routes_df["route_no"],
            routes_df["required_energy_kwh"]
        )
    )


    # ============================================================
    # Route Activity
    # ============================================================

    route_active = {}

    for r in routes:

        departure = route_departure[r]
        return_time = routes_ending[r]

        for t in times:

            if departure <= t < return_time:

                route_active[(r, t)] = 1

            else:

                route_active[(r, t)] = 0


    # ============================================================
    # Route Departure Timestep
    # ============================================================

    # Departure is already an exact hourly timestamp,
    # so no further rounding is required.

    route_departure_t = {
        r: route_departure[r]
        for r in routes
    }


    # ============================================================
    # Candidate Batteries / Chargers
    # ============================================================

    batteries = list(
        range(1, len(routes_df) + 1)
    )

    chargers = list(
        range(1, len(routes_df) + 1)
    )


    # ============================================================
    # Return Processed Data
    # ============================================================

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
        route_departure_t,
        batteries,
        chargers,
    )