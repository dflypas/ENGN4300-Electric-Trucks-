# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:47:59 2026

@author: d_par
"""
import os
import pandas as pd
import nemreader as nr
import polars as pl

os.chdir(r"C:\Users\d_par\OneDrive\Desktop\Danu\Uni\Capstone")

target_csv = "NEM12_DNSP_Data_Dairy_Drive.csv"

# Parse the NEM12 file
nem_file = target_csv
mms_data = nr.NEMFile(nem_file)


# Get data as a Pandas DataFrame (long or wide format)
df = mms_data.get_data_frame_long()  # or wide=False for long format
df = df.to_pandas()

ar_suffix = df['suffix'].unique()

df_B1 = df[df['suffix'] == 'B1'].copy()
df_E1 = df[df['suffix'] == 'E1'].copy()
df_K1 = df[df['suffix'] == 'K1'].copy()
df_Q1 = df[df['suffix'] == 'Q1'].copy()