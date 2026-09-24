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
import optimiser as opt
import route_data_processor as rt
import business_as_usual as bau
import Optimiser_second as opt2

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

    input_data = pd.read_csv(os.path.join(INPUT_DIR, "input_data.csv"))
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

    YEAR = float(inputs["year"]) 
    

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
    FIT_on_peak = float(inputs["F_I_T_on_peak"])
    FIT_off_peak = float(inputs["F_I_T_off_peak"])
    FIT_flat = float(inputs["F_I_T_flat"])
    WM = float(inputs["Wholesale Market"])

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
    DIESEL_L = float(inputs['DIESEL_L'])

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
        times=TIMES
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
    ROUTE_DIESEL_USED = route_data_processor.route_diesel_used
    ACTIVE_ROUTES_AT = route_data_processor.active_routes_at
    ROUTES_DEPARTING_AT = route_data_processor.routes_departing_at
    OVERLAPPING_ROUTES = route_data_processor.overlapping_routes

    TRUCKS_IN_YEAR = {y: TRUCKS for y in YEARS}
    FLEET_SIZE = {y: len(TRUCKS) for y in YEARS}
    TRUCK_FIRST_YEAR = {i: YEARS[0] for i in TRUCKS}
    TRUCK_BASE_ELECTRIC = {i: 0 for i in TRUCKS}

    edp = ep.EnergyDataProcessor(ASSESSMENT_YEAR, LATITUDE, LONGITUDE)
    df_data = edp.build_time_index(NUMBER_OF_TIMESTEPS, DT)

    GRID_PRICE_T = dict(zip(TIMES, edp.get_grid_price_profile(
        df_data, GRID_ONPEAK_PRICE, GRID_OFFPEAK_PRICE)))

    FEED_IN_TARIFF_T = dict(zip(TIMES, edp.get_feed_in_tariff_profile(
        df_data, FIT_on_peak, FIT_off_peak, FIT_flat)))

    df_data, MONTH_LIMITS, NETWORK_ACCESS_COST = edp.get_network_tariff_data(
        df_data, ep.NETWORK_TARIFF, DT)
    M = len(MONTH_LIMITS)

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
        feed_in_tariff_t=FEED_IN_TARIFF_T,
        whole_market = WM,
        month_limits = MONTH_LIMITS,
        months = M,
        dict_limits = ep.NETWORK_TARIFF,
        network_access_cost = NETWORK_ACCESS_COST,
        df_data = df_data,


        route_diesel_used = ROUTE_DIESEL_USED,
        diesel_price_t=DIESEL_PRICE_T,

        duration_days=DURATION_DAYS,
        dt=DT,
        years=YEARS,
        first_t=FIRST_T,
        last_t=LAST_T,

        batteries=BATTERIES,
        chargers=CHARGERS,

        n_trucks=N_TRUCKS,

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
        diesel_lifetime=DIESEL_L,

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

    if df_results is not None:
        df_results.to_csv(os.path.join(INPUT_DIR, "optimal_results.csv"), index=False)
    else:
        raise RuntimeError("Year 1 found no feasible solution, so there is nothing to feed into year 2.")
    
    # ============================================================
    # Run Business as Usual
    # ============================================================

    bau_results, bau_df_results = bau.business_as_usual(

        routes=ROUTES,
        route_diesel_used=ROUTE_DIESEL_USED,
        route_year=ROUTE_YEAR,
        diesel_price_t=DIESEL_PRICE_T,

        trucks=TRUCKS,
        years=YEARS,
        trucks_in_year=TRUCKS_IN_YEAR,

        diesel_capex=DIESEL_CAPEX,
        diesel_lifetime=DIESEL_L,
        diesel_om=DIESEL_OM,

        discount_rate=DISCOUNT_RATE,
        project_lifetime=N_YEARS,
    )

    # ====================================
    # Optimiser Year 2
    # ====================================

    partial_data = pd.read_csv(
    os.path.join(INPUT_DIR, "optimal_results.csv")
    )

    optimal = partial_data.iloc[0].to_dict() #Necessary for the way for reading the data in the structure it is outputted

    # Time Set
    DURATION_DAYS = int(inputs["duration"])
    N_YEARS = max(1, math.ceil(DURATION_DAYS / 365))
    YEARS = list(range(N_YEARS))
    ASSESSMENT_YEAR = int(inputs["year"]) + 1 #Literally just adding one for the next year
    DT = float(inputs["dt"])

    YEAR = float(inputs["year"]) + 1 #Literally just adding one for the next year
    

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

    # Candidates
    MAX_CANDIDATE_BATTERIES = int(inputs["max_candidate_batteries"])
    MIN_CANDIDATE_BATTERIES = int(optimal["number_of_batteries"]) #Minimum battery numbers (not actually necessary but don't want to delete)
    MAX_CANDIDATE_CHARGERS = int(inputs["max_candidate_chargers"]) 
    MIN_CANDIDATE_CHARGERS = int(optimal["number_of_chargers"]) #Minimum charger numbers (not actually necessary but don't want to delete)

    BATTERIES = list(range(MAX_CANDIDATE_BATTERIES))
    CHARGERS  = list(range(MAX_CANDIDATE_CHARGERS))

    # Batteries
    SB_CAP_MAX = float(inputs["SB_cap_max"])
    SB_CAP_MIN = float(optimal["number_of_batteries"]) #NEW VARIABLE (current charger bought) - TODo Rename this. It is very confusing.
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
    CHARGER_MIN = int(optimal["number_of_chargers"]) #Minimum amount of chargers

    # BESS
    BESS_INITIAL_SOC = float(inputs["BESS_initial_soc"])
    BESS_MIN_SOC = float(inputs["BESS_min_soc"])
    BESS_MAX_SOC = float(inputs["BESS_max_soc"])

    BESS_CAP_MAX = float(inputs["BESS_cap_max"])
    BESS_CAP_MIN = float(optimal["BESS_capacity_kWh"]) #Minimum installed BESS
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
    FIT_on_peak = float(inputs["F_I_T_on_peak"])
    FIT_off_peak = float(inputs["F_I_T_off_peak"])
    FIT_flat = float(inputs["F_I_T_flat"])
    WM = float(inputs["Wholesale Market"])

    # PV
    PV_EFFICIENCY = float(inputs["PV_efficiency"])
    PV_MAX_KW = float(inputs["PV_max_kw"])
    PV_MIN_KW = float(optimal["PV_capacity_kW"]) #Already installed PV

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
    DIESEL_L = float(inputs['DIESEL_L'])

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
    spot_price_df = spot_price_df[spot_price_df["year"] >= ASSESSMENT_YEAR]
    spot_values = spot_price_df["Spot Price"].tolist()
    SPOT_PRICE_T = dict(zip(TIMES, spot_values[:len(TIMES)]))


    route_data_processor = rt.RouteDataProcessor(
        route_csv=os.path.join(INPUT_DIR, "route_schedule.csv"),
        duration_days=DURATION_DAYS,
        times=TIMES
    )

    # Diesel Price Reading: offset by how many years after the first forecast year we are
    FIRST_FORECAST_YEAR = 2026          # first year in the spot file
    YEAR_IDX = ASSESSMENT_YEAR - FIRST_FORECAST_YEAR   # 0 for 2026, 1 for 2027...
    diesel_values = diesel_price_df["Price"].tolist()
    DIESEL_PRICE_T = dict(zip(YEARS, diesel_values[YEAR_IDX:YEAR_IDX + len(YEARS)]))

    ROUTES = route_data_processor.routes
    TRUCKS = route_data_processor.trucks
    ROUTE_TRUCK = route_data_processor.route_truck
    ROUTE_YEAR = route_data_processor.route_year
    ROUTE_DEPARTURE = route_data_processor.route_departure
    ROUTE_RETURN = route_data_processor.route_return
    ROUTE_PRE_DEPARTURE = route_data_processor.route_pre_departure
    ROUTE_ENERGY = route_data_processor.route_energy
    ROUTE_DIESEL_USED = route_data_processor.route_diesel_used
    ACTIVE_ROUTES_AT = route_data_processor.active_routes_at
    ROUTES_DEPARTING_AT = route_data_processor.routes_departing_at
    OVERLAPPING_ROUTES = route_data_processor.overlapping_routes



    TRUCKS_IN_YEAR = {y: TRUCKS for y in YEARS}
    FLEET_SIZE = {y: len(TRUCKS) for y in YEARS}
    TRUCK_FIRST_YEAR = {i: YEARS[0] for i in TRUCKS}
    N_BASE = round(float(optimal["number_of_retrofits"]))
    TRUCK_BASE_ELECTRIC = {i: 1 if idx < N_BASE else 0 for idx, i in enumerate(TRUCKS)}  #Setting up the already retrofitted trucks

    edp = ep.EnergyDataProcessor(ASSESSMENT_YEAR, LATITUDE, LONGITUDE)
    df_data = edp.build_time_index(NUMBER_OF_TIMESTEPS, DT)

    GRID_PRICE_T = dict(zip(TIMES, edp.get_grid_price_profile(
        df_data, GRID_ONPEAK_PRICE, GRID_OFFPEAK_PRICE)))

    FEED_IN_TARIFF_T = dict(zip(TIMES, edp.get_feed_in_tariff_profile(
        df_data, FIT_on_peak, FIT_off_peak, FIT_flat)))

    df_data, MONTH_LIMITS, NETWORK_ACCESS_COST = edp.get_network_tariff_data(
        df_data, ep.NETWORK_TARIFF, DT)
    M = len(MONTH_LIMITS)

    results_year, df_results_year = opt2.optimiser(
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
        feed_in_tariff_t=FEED_IN_TARIFF_T,
        whole_market = WM,
        month_limits = MONTH_LIMITS,
        months = M,
        dict_limits = ep.NETWORK_TARIFF,
        network_access_cost = NETWORK_ACCESS_COST,
        df_data = df_data,


        route_diesel_used = ROUTE_DIESEL_USED,
        diesel_price_t=DIESEL_PRICE_T,

        duration_days=DURATION_DAYS,
        dt=DT,
        years=YEARS,
        first_t=FIRST_T,
        last_t=LAST_T,

        batteries=BATTERIES,
        chargers=CHARGERS,

        n_trucks=N_TRUCKS,

        sb_cap=SB_CAP_MAX,
        sb_cap_min= SB_CAP_MIN,
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
        charger_min= CHARGER_MIN,

        bess_initial_soc=BESS_INITIAL_SOC,
        bess_min_soc=BESS_MIN_SOC,
        bess_max_soc=BESS_MAX_SOC,
        bess_max_cap=BESS_CAP_MAX,
        bess_min_cap= BESS_CAP_MIN,
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
        pv_min = PV_MIN_KW,
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
        diesel_lifetime=DIESEL_L,

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
        truck_base_electric= TRUCK_BASE_ELECTRIC,

        routes_departing_at=ROUTES_DEPARTING_AT,
        route_pre_departure=ROUTE_PRE_DEPARTURE,

        aux_load_t=AUX_LOAD_T,

        time_limit=time_limit,
        mip_gap=mip_gap,
    )

    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    if df_results_year is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_results_year.to_csv(
            os.path.join(OUTPUT_DIR, "results_year2.csv"),
            index=False,
        )
    else:
        print("No feasible solution, nothing saved.")


    return results, df_results, bau_results, bau_df_results, results_year, df_results_year #results, df_results



# Allow running this file directly (python main.py) exactly as before,
# with no UI overrides and no time limit.
if __name__ == "__main__":
    run_pipeline()
