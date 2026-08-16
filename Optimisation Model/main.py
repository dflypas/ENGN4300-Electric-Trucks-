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
# Process and Define Static Input Data Names
# ============================================================

# Read input file
input_data = pd.read_csv("input_data.csv")
input_values = input_data.set_index("field_name")["value"].to_dict()

# Location inputs
ASSESSMENT_YEAR = int(input_values["year"])
LATITUDE = float(input_values["latitude"])
LONGITUDE = float(input_values["longitude"])

# PV inputs
pv_efficiency = float(input_values["pv_panel_efficiency"])
max_pv_area = float(input_values["max_panel_area"])

# Grid inputs
peak_grid_electricity_price = float(input_values["peak_grid_electricity_price"])
offpeak_grid_electricity_price = float(input_values["offpeak_grid_electricity_price"])
feed_in_tariff = float(input_values["feed_in_tariff"])
max_cap_grid_import = float(input_values["max_grid_import_capacity"])
max_cap_grid_export = float(input_values["max_grid_export_capacity"])

# Battery inputs
max_cap_sb = float(input_values["battery_capacity"])
sb_charge_efficiency = float(input_values["sb_charge_efficiency"])








# ============================================================
# Process Data
# ============================================================

CALC_INPUT_DATA = True

# Hourly input data
if CALC_INPUT_DATA:
    # Initialize and process input data
    data_processor = ep.EnergyDataProcessor(ASSESSMENT_YEAR, LATITUDE, LONGITUDE)








# ============================================================
# Base Optimisation Scenarios
# ============================================================

opt.optimiser()

