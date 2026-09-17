#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This doc is using Viv's Structure
main.py is the main entry point for the DairyDrive Electric optimisation pipeline. It loads input data, applies any overrides, processes data, and runs the optimisation model.
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

# Import Own Modules
import energy_data_processor as ep
import updated_optimiser as opt
import route_data_processor as rt

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Cached helper: PV profile generation
# ============================================================

@lru_cache(maxsize=8)
def _get_pv_profile(assessment_year, latitude, longitude):
    energy_data_processor = ep.EnergyDataProcessor(assessment_year, latitude, longitude)
    pv_generated_df = energy_data_processor.create_pv_profile_csv("pv_profile_output.csv")
    return tuple(pv_generated_df["PV_Profile"])  # tuple so it's hashable/cacheable


def run_pipeline(overrides=None, time_limit=None, mip_gap=0.05):
    """
    Runs the full DairyDrive Electric pipeline: loads CSV defaults,
    applies any UI-supplied overrides, processes data, builds and
    solves the Gurobi model, and returns (results, df_results).

    overrides: dict of {field_name: value} to override CSV defaults.
               Keys must match the 'field_name' column in
               input_data_user_naming.csv, e.g.:
               {"max_candidate_batteries": 10, "PV_max_kw": 6000, ...}
    """

    # ============================================================
    # Load and Define Static Input Data
    # ============================================================

    input_data = pd.read_csv(os.path.join(INPUT_DIR, "input_data_user_naming.csv"))
    inputs = input_data.set_index("field_name")["value"].to_dict()

    # Apply UI overrides on top of CSV defaults
    if overrides:
        inputs.update(overrides)

    # Time Set
    DURATION_DAYS = int(inputs["duration"])
    N_YEARS = max(1, math.ceil(DURATION_DAYS / 365))
    YEARS = list(range(N_YEARS))
    ASSESSMENT_YEAR = int(inputs["year"])
    DT = float(inputs["dt"])

    NUMBER_OF_TIMESTEPS = int(DURATION_DAYS * 24 / DT)
    TIMES = list(range(NUMBER_OF_TIMESTEPS))
    FIRST_T = TIMES[0]
    LAST_T = TIMES[-1]
    PREV_T = {TIMES[i]: TIMES[i - 1] for i in range(1, len(TIMES))}

    # Location
    LATITUDE = float(inputs["latitude"])
    LONGITUDE = float(inputs["longitude"])

    # Fleet Inputs and Diesel
    N_TRUCKS = int(inputs["n_trucks"])
    SB_TRUCK_CONSUMPTION = float(inputs["SB_truck_consumption"])
    DIESEL_TRUCK_CONSUMPTION = float(inputs["DIESEL_truck_consumption"])
    DIESEL_COST_PER_KWH = float(inputs["DIESEL_cost_per_kwh"])

    # Candidates
    MAX_CANDIDATE_BATTERIES = int(inputs["max_candidate_batteries"])
    MAX_CANDIDATE_CHARGERS = int(inputs["max_candidate_chargers"])

    BATTERIES = list(range(MAX_CANDIDATE_BATTERIES))
    CHARGERS = list(range(MAX_CANDIDATE_CHARGERS))

    # Batteries
    SB_CAP_MAX = float(inputs["SB_cap_max"])
    SOC_SB_MIN = float(inputs["SOC_SB_min"])
    SOC_SB_MAX = float(inputs["SOC_SB_max"])
    SOC_SB_0 = float(inputs["SOC_SB_0"])

    SB_EFFICIENCY_CHARGE = float(inputs["SB_efficiency_charge"])
    SB_EFFICIENCY_DISCHARGE = float(inputs["SB_efficiency_discharge"])
    SB_SELF_DISCHARGE = float(inputs["SB_self_discharge"])
    SB_MAX_CHARGE_POWER = float(inputs["SB_max_charge_power"])

    E_SB_MIN = SOC_SB_MIN * SB_CAP_MAX
    E_SB_MAX = SOC_SB_MAX * SB_CAP_MAX
    E_SB_INITIAL = SOC_SB_0 * SB_CAP_MAX

    # Charger
    C_CHARGE = float(inputs["C_charge"])

    # BESS
    BESS_INITIAL_SOC = float(inputs["BESS_initial_soc"])
    BESS_MIN_SOC = float(inputs["BESS_min_soc"])
    BESS_MAX_SOC = float(inputs["BESS_max_soc"])

    BESS_CAP_MAX = float(inputs["BESS_cap_max"])
    BESS_MAX_CHARGE_POWER = float(inputs["BESS_charge"])
    BESS_MAX_DISCHARGE_POWER = float(inputs["BESS_discharge"])

    BESS_EFFICIENCY_CHARGE = float(inputs["BESS_efficiency_charge"])
    BESS_EFFICIENCY_DISCHARGE = float(inputs["BESS_efficiency_discharge"])
    BESS_SELF_DISCHARGE = float(inputs["BESS_self_discharge"])

    BESS_CHARGE_C_RATE = float(inputs["BESS_charge_c_rate"])
    BESS_DISCHARGE_C_RATE = float(inputs["BESS_discharge_c_rate"])

    # Grid
    GRID_MAX_IMPORT = float(inputs["GRID_max_import"])
    GRID_MAX_EXPORT = float(inputs["GRID_max_export"])
    GRID_EXISTING_CAPACITY = float(inputs["GRID_existing_capacity"])
    GRID_MAX_CAPACITY = float(inputs["GRID_max_capacity"])
    GRID_ONPEAK_PRICE = float(inputs["GRID_onpeak_price"])
    GRID_OFFPEAK_PRICE = float(inputs["GRID_offpeak_price"])

    # PV
    PV_EFFICIENCY = float(inputs["PV_efficiency"])
    PV_MAX_KW = float(inputs["PV_max_kw"])

    # CAPEX
    C_CAPEX = float(inputs["C_CAPEX"])
    SB_CAPEX = float(inputs["SB_CAPEX"])
    PV_CAPEX_PER_KW = float(inputs["PV_CAPEX_per_kw"])
    BESS_CAPEX_PER_KW = float(inputs["BESS_CAPEX_per_kw"])
    RETROFIT_CAPEX = float(inputs["RETROFIT_CAPEX"])
    DIESEL_CAPEX = float(inputs["DIESEL_CAPEX"])

    # LIFETIMES
    SB_L = float(inputs["SB_L"])
    C_L = float(inputs["C_L"])
    BESS_L = float(inputs["BESS_L"])
    PV_L = float(inputs["PV_L"])
    RETROFIT_L = float(inputs["RETROFIT_L"])

    # O & M
    SB_OM = float(inputs["SB_OM"])
    C_OM = float(inputs["C_OM"])
    BESS_OM = float(inputs["BESS_OM"])
    PV_OM = float(inputs["PV_OM"])
    RETROFIT_OM = float(inputs["RETROFIT_OM"])
    DIESEL_OM = float(inputs["DIESEL_OM"])

    # OTHER
    DISCOUNT_RATE = float(inputs["discount_rate"])
    AUX_LOAD_T = {t: 0 for t in TIMES}

    # ============================================================
    # Process Data
    # ============================================================

    # PV profile — cached; only recomputed if year/lat/lon change
    pv_profiles = _get_pv_profile(ASSESSMENT_YEAR, LATITUDE, LONGITUDE)
    PV_PROFILES_T = dict(zip(TIMES, pv_profiles[:len(TIMES)]))

    spot_price_df = pd.read_csv(os.path.join(INPUT_DIR, "spot_price.csv"))
    spot_values = list(spot_price_df["Spot Price"])
    SPOT_PRICE_T = dict(zip(TIMES, spot_values[:len(TIMES)]))
    SELL_PRICE_T = {time: 10 for time in TIMES}

    route_data_processor = rt.RouteDataProcessor(
        route_csv=os.path.join(INPUT_DIR, "route_schedule.csv"),
        duration_days=DURATION_DAYS,
        times=TIMES,
        truck_consumption=SB_TRUCK_CONSUMPTION
    )

    diesel_price_df = pd.read_csv(os.path.join(INPUT_DIR, "Diesel Price Prediction.csv"))
    diesel_values = diesel_price_df["Price"].tolist()

    DIESEL_PRICE_T = dict(zip(YEARS, diesel_values[:len(YEARS)]))

    ROUTES = route_data_processor.routes
    TRUCKS = route_data_processor.trucks
    ROUTE_TRUCK = route_data_processor.route_truck
    ROUTE_YEAR = route_data_processor.route_year
    ROUTE_DEPARTURE = route_data_processor.route_departure
    ROUTE_RETURN = route_data_processor.route_return
    ROUTE_PRE_DEPARTURE = route_data_processor.route_pre_departure
    ROUTE_ENERGY = route_data_processor.route_energy
    ACTIVE_ROUTES_AT = route_data_processor.active_routes_at
    ROUTES_DEPARTING_AT = route_data_processor.routes_departing_at
    OVERLAPPING_ROUTES = route_data_processor.overlapping_routes

    TRUCKS_IN_YEAR = {y: TRUCKS for y in YEARS}
    FLEET_SIZE = {y: len(TRUCKS) for y in YEARS}
    TRUCK_FIRST_YEAR = {i: YEARS[0] for i in TRUCKS}
    TRUCK_BASE_ELECTRIC = {i: 0 for i in TRUCKS}

    GRID_PRICE_T = {}
    for t in TIMES:
        hour_of_day = t % 24
        if 15 <= hour_of_day < 21:
            GRID_PRICE_T[t] = GRID_ONPEAK_PRICE
        else:
            GRID_PRICE_T[t] = GRID_OFFPEAK_PRICE

    # ============================================================
    # Run Optimisation
    # ============================================================

    results, df_results = opt.optimiser(
        routes=ROUTES,
        trucks=TRUCKS,
        times=TIMES,

        route_truck=ROUTE_TRUCK,
        active_routes_at=ACTIVE_ROUTES_AT,
        route_energy=ROUTE_ENERGY,
        route_departure=ROUTE_DEPARTURE,
        route_return=ROUTE_RETURN,
        overlapping_routes=OVERLAPPING_ROUTES,

        pv_profiles_t=PV_PROFILES_T,
        spot_price_t=SPOT_PRICE_T,
        sell_price_t=SELL_PRICE_T,
        grid_price_t=GRID_PRICE_T,

        diesel_truck_consumption=DIESEL_TRUCK_CONSUMPTION,
        diesel_cost_per_kwh=DIESEL_COST_PER_KWH,
        diesel_price_t=DIESEL_PRICE_T,

        duration_days=DURATION_DAYS,
        dt=DT,
        years=YEARS,
        first_t=FIRST_T,
        last_t=LAST_T,

        batteries=BATTERIES,
        chargers=CHARGERS,

        n_trucks=N_TRUCKS,
        truck_consumption=SB_TRUCK_CONSUMPTION,

        sb_cap=SB_CAP_MAX,
        sb_min_soc=SOC_SB_MIN,
        sb_max_soc=SOC_SB_MAX,
        sb_initial_soc=SOC_SB_0,
        sb_charge_efficiency=SB_EFFICIENCY_CHARGE,
        sb_discharge_efficiency=SB_EFFICIENCY_DISCHARGE,
        sb_self_discharge=SB_SELF_DISCHARGE,
        sb_max_charge_power=SB_MAX_CHARGE_POWER,

        battery_cost=SB_CAPEX,
        charger_power_kw=C_CHARGE,
        charger_cost=C_CAPEX,

        bess_initial_soc=BESS_INITIAL_SOC,
        bess_min_soc=BESS_MIN_SOC,
        bess_max_soc=BESS_MAX_SOC,
        bess_max_cap=BESS_CAP_MAX,
        bess_max_charge_power=BESS_MAX_CHARGE_POWER,
        bess_max_discharge_power=BESS_MAX_DISCHARGE_POWER,
        bess_charge_efficiency=BESS_EFFICIENCY_CHARGE,
        bess_discharge_efficiency=BESS_EFFICIENCY_DISCHARGE,
        bess_self_discharge=BESS_SELF_DISCHARGE,
        bess_charge_c_rate=BESS_CHARGE_C_RATE,
        bess_discharge_c_rate=BESS_DISCHARGE_C_RATE,

        max_grid_import=GRID_MAX_IMPORT,
        max_grid_export=GRID_MAX_EXPORT,
        existing_grid_capacity=GRID_EXISTING_CAPACITY,
        max_grid_capacity=GRID_MAX_CAPACITY,

        pv_efficiency=PV_EFFICIENCY,
        pv_max_kw=PV_MAX_KW,
        pv_capex_per_kw=PV_CAPEX_PER_KW,

        bess_capex_per_kwh=BESS_CAPEX_PER_KW,
        retrofit_cost=RETROFIT_CAPEX,
        diesel_capex=DIESEL_CAPEX,

        discount_rate=DISCOUNT_RATE,
        project_lifetime=N_YEARS,

        sb_lifetime=SB_L,
        charger_lifetime=C_L,
        bess_lifetime=BESS_L,
        pv_lifetime=PV_L,
        retrofit_lifetime=RETROFIT_L,

        sb_om=SB_OM,
        charger_om=C_OM,
        bess_om=BESS_OM,
        pv_om=PV_OM,
        retrofit_om=RETROFIT_OM,
        diesel_om=DIESEL_OM,

        route_year=ROUTE_YEAR,
        trucks_in_year=TRUCKS_IN_YEAR,
        fleet_size=FLEET_SIZE,
        truck_first_year=TRUCK_FIRST_YEAR,
        truck_base_electric=TRUCK_BASE_ELECTRIC,

        routes_departing_at=ROUTES_DEPARTING_AT,
        route_pre_departure=ROUTE_PRE_DEPARTURE,

        aux_load_t=AUX_LOAD_T,

        time_limit=time_limit,
        mip_gap=mip_gap,
    )

    return results, df_results


# Allow running this file directly (python main.py) exactly as before,
# with no UI overrides and no time limit.
if __name__ == "__main__":
    run_pipeline()
