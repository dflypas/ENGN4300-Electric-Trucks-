#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:19:24 2026

@author: mollyrobertson
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import ast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

df = pd.read_csv(os.path.join(RESULTS_DIR, "results_year_timeseries2026.csv"))
df_plot = df.iloc[:168]

# ============================================================
# 1. Swappable Battery Charging and Depletion
# ============================================================

battery_energy_cols = [col for col in df_plot.columns if col.startswith("SB_") and col.endswith("_Energy_kWh")]

plt.figure(figsize=(12, 6))

for col in battery_energy_cols:
    plt.plot(df_plot["Time"], df_plot[col], label=col.replace("_Energy_kWh", ""))

plt.xlabel("Time (hours)")
plt.ylabel("Battery Energy (kWh)")
plt.title("Swappable Battery Charging and Depletion")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# 2. Route Schedule
# ============================================================

df_plot["Active_Routes"] = df_plot["Active_Routes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_plot["Route_Batteries"] = df_plot["Route_Batteries"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

plt.figure(figsize=(12, 6))

for idx, row in df_plot.iterrows():
    for route in row["Active_Routes"]:
        plt.scatter(row["Time"], route, s=5)

plt.xlabel("Time (hours)")
plt.ylabel("Route Number")
plt.title("Route Schedule")
plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# 3. PV, Grid and BESS Operation
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(df_plot["Time"], df_plot["PV_Generated_kW"], label="PV Generation")
plt.plot(df_plot["Time"], df_plot["Grid_Import_kW"], label="Grid Import")
plt.plot(df_plot["Time"], df_plot["BESS_Discharge_kW"], label="BESS Discharge")
plt.plot(df_plot["Time"], -df_plot["BESS_Charge_kW"], label="BESS Charge")

plt.xlabel("Time (hours)")
plt.ylabel("Power (kW)")
plt.title("PV Generation, Grid Import and BESS Operation")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# 4. BESS Storage
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(df_plot["Time"], df_plot["BESS_Energy_kWh"])

plt.xlabel("Time (hours)")
plt.ylabel("Stored Energy (kWh)")
plt.title("BESS Energy Storage")
plt.grid()
plt.tight_layout()
plt.show()