#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This doc is using Viv's Structure

"""

import json
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


# ============================================================
# Load and Define Static Input Dat
# ============================================================

# Read input files
input_data = pd.read_csv("input_data.csv")
inputs = input_data.set_index("field_name")["value"].to_dict()


# Create Timeset
DURATION_DAYS = int(inputs["duration"])
DT = float(inputs["time_step_dt"])
NUMBER_OF_HOURS = DURATION_DAYS * 24
TIMES = list(range(NUMBER_OF_HOURS))
FIRST_T = TIMES[0]
LAST_T = TIMES[-1]
PREV_T = {TIMES[i]: TIMES[i - 1] for i in range(1, len(TIMES))}


# Location inputs 
ASSESSMENT_YEAR = int(inputs["year"])
LATITUDE = float(inputs["latitude"])
LONGITUDE = float(inputs["longitude"])

# PV
PV_CAPEX_PER_KW = float(inputs["pv_capex_per_kw"])

# Grid
FEED_IN_TARIFF = float(inputs["feed_in_tariff"])
max_grid_import = float(inputs["max_grid_import_capacity"])
max_grid_export = float(inputs["max_grid_export_capacity"])

# grid_electricity_price and existing_grid_capacity_kw are not yet in
# input_data.csv. Using .get with a fallback so this keeps running until
# those rows are added - add them for a real (non-zero) grid cost result.
GRID_ELECTRICITY_PRICE = float(inputs.get("grid_electricity_price", 0.0))
EXISTING_GRID_CAPACITY_KW = float(inputs.get("existing_grid_capacity_kw", 0.0))

# Swappable Battery / Charger
TRUCK_CONSUMPTION = float(inputs["truck_consumption"])
MAX_CANDIDATE_BATTERIES = int(inputs["max_candidate_batteries"])
BATTERIES = list(range(1, MAX_CANDIDATE_BATTERIES + 1))
sb_cap = float(inputs['sb_max_capacity'])
sb_min_soc = float(inputs['sb_min_soc'])
sb_max_soc = float(inputs['sb_max_soc'])
sb_initial_soc = float(inputs['sb_initial_soc'])
charge_efficiency = float(inputs['sb_charge_efficiency'])
discharge_efficiency = float(inputs['sb_discharge_efficiency'])
BATTERY_COST = float(inputs['battery_cost'])
charger_power_kw = float(inputs['charger_power'])
CHARGER_COST = float(inputs['charger_cost'])
    
E_SB_min = sb_min_soc * sb_cap
E_SB_max = sb_max_soc * sb_cap
E_SB_initial = sb_initial_soc * sb_cap
max_sb_charge_per_step = charger_power_kw * DT
    
# BESS
soc_BESS_initial = float(inputs['BESS_initial_soc'])
BESS_min_soc = float(inputs['BESS_min_soc'])
BESS_max_soc = float(inputs['BESS_max_soc'])
BESS_CAPEX_PER_KWH = float(inputs['bess_capex_per_kwh'])
BESS_charge_efficiency = float(inputs['bess_charge_efficiency'])
BESS_discharge_efficiency = float(inputs['bess_discharge_efficiency'])
BESS_charge_c_rate = float(inputs['bess_charge_c_rate'])
BESS_discharge_c_rate = float(inputs['bess_discharge_c_rate'])

# bess_max_capacity_kwh is not yet in input_data.csv. Falls back to a large
# default (effectively unconstrained) until a real site limit is added.
BESS_MAX_CAPACITY_KWH = float(inputs.get('bess_max_capacity_kwh', 1e6))
    
# Retrofit
RETRO_COST = float(inputs["retrofit_cost_per_truck"])
n_total_trucks = int(inputs["n_total_trucks"])
diesel_fleet_annual_km = float(inputs["diesel_fleet_annual_km"])
diesel_price_per_km = float(inputs["diesel_price_per_km"])
    
# Discount, Useful Life
DISCOUNT_RATE = float(inputs['discount_rate'])
PROJECT_LT = float(inputs['project_lifetime'])
SB_LT = float(inputs['sb_lifetime'])
CS_LT = float(inputs['charger_lifetime'])
BESS_LT = float(inputs['bess_lifetime'])
PV_LT = float(inputs['pv_lifetime'])
RETRO_LT = float(inputs['retrofit_lifetime'])
    
# Maintenance
CM_SB = float(inputs['sb_maintenance_rate'])
CM_CS = float(inputs['charger_maintenance_rate'])
CM_BESS = float(inputs['bess_maintenance_rate'])
CM_PV = float(inputs['pv_maintenance_rate'])
CM_RETRO = float(inputs['retrofit_maintenance_rate'])


# ============================================================
# Process Data
# ============================================================

CALC_INPUT_DATA = True

if CALC_INPUT_DATA:
    # Process Energy Data
    energy_data_processor = ep.EnergyDataProcessor(ASSESSMENT_YEAR, LATITUDE, LONGITUDE)
    
    pv_generated_df = (energy_data_processor.create_pv_profile_csv("pv_profile_output.csv"))
    pv_values = list(pv_generated_df["PV_Profile"] * 1000)
    PV_GENERATED_T = dict(zip(TIMES, pv_values[:len(TIMES)]))
    
    # Process Route Data
    route_data_processor = rt.RouteDataProcessor(route_csv="route_schedule.csv", duration_days=DURATION_DAYS, times=TIMES)

    ROUTES_DF = route_data_processor.routes_df
    ROUTES = route_data_processor.routes
    ROUTE_DEPARTURE = route_data_processor.route_departure
    ROUTE_RETURN = route_data_processor.route_return
    ACTIVE_ROUTES_AT = route_data_processor.active_routes_at

    # Flat grid import/export network limits applied at every timestep.
    # Swap for a real per-timestep profile if network limits vary by hour.
    GRID_IMPORT_MAX_T = {t: max_grid_import for t in TIMES}
    GRID_EXPORT_MAX_T = {t: max_grid_export for t in TIMES}



# ============================================================
# Base Optimisation Scenarios
# ============================================================


opt.optimiser(
    routes=ROUTES,
    times=TIMES,
    first_t=FIRST_T,
    prev_t=PREV_T,
    batteries=BATTERIES,
    active_routes_at=ACTIVE_ROUTES_AT,
    truck_consumption=TRUCK_CONSUMPTION,
    dt=DT,
    sb_cap=sb_cap,
    E_SB_min=E_SB_min,
    E_SB_max=E_SB_max,
    E_SB_initial=E_SB_initial,
    charge_efficiency=charge_efficiency,
    discharge_efficiency=discharge_efficiency,
    battery_cost=BATTERY_COST,
    charger_power_kw=charger_power_kw,
    charger_cost=CHARGER_COST,
    max_sb_charge_per_step=max_sb_charge_per_step,
    soc_BESS_initial=soc_BESS_initial,
    pv_generated_t=PV_GENERATED_T,
    bess_min_soc=BESS_min_soc,
    bess_max_soc=BESS_max_soc,
    bess_capex_per_kwh=BESS_CAPEX_PER_KWH,
    bess_charge_efficiency=BESS_charge_efficiency,
    bess_discharge_efficiency=BESS_discharge_efficiency,
    bess_charge_c_rate=BESS_charge_c_rate,
    bess_discharge_c_rate=BESS_discharge_c_rate,
    bess_max_capacity_kwh=BESS_MAX_CAPACITY_KWH,
    grid_electricity_price=GRID_ELECTRICITY_PRICE,
    feed_in_tariff=FEED_IN_TARIFF,
    grid_existing_kw=EXISTING_GRID_CAPACITY_KW,
    grid_max_kw=max_grid_import,
    grid_import_max=GRID_IMPORT_MAX_T,
    grid_export_max=GRID_EXPORT_MAX_T,
)