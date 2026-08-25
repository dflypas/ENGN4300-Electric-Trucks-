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

