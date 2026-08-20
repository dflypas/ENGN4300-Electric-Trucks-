# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:47:59 2026

@author: d_par
"""
import os
import pandas as pd
import nemreader as nr
import polars as pl

# Set Working directory
os.chdir(r"C:\Users\d_par\OneDrive\Desktop\Danu\Uni\Capstone")

# Name of NEM12 CSV to be read
target_csv = "NEM12_DNSP_Data_Dairy_Drive.csv"

# Read the NEM12 CSV asa Polars DF using nemreader
nem_file = target_csv
mms_data = nr.NEMFile(nem_file)


# Convert dataframe to a Pandas DataFrame
df = mms_data.get_data_frame_long()
df = df.to_pandas()

# Read all type suffix's from the DF
# For this model it will be 4 'default'suffixes: B1, E1, K1, Q1
ar_suffix = df['suffix'].unique()

# Seperate into unique DF based on Suffix
df_B1 = df[df['suffix'] == 'B1'].copy()
df_E1 = df[df['suffix'] == 'E1'].copy()
df_K1 = df[df['suffix'] == 'K1'].copy()
df_Q1 = df[df['suffix'] == 'Q1'].copy()

