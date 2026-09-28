#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:42:58 2026

@author: mollyrobertson
"""

import os
import pandas as pd
import numpy as np

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from energy_data_processor import EnergyDataProcessor
from energy_data_processor import pv_profile_generator_tmy

def test_get_weather_data():
    
    EDP = EnergyDataProcessor(year=2026, lat=-36.67, lon=149.84)
    weather_data, elevation = EDP.get_weather_data()
    
    # Is the weather data stucture in the expected layout and columns
    required_columns = ["temp_air",
        "wind_speed",
        "ghi",
        "dni",
        "dhi",
        "pressure"]
    for column in required_columns:
        assert column in weather_data.columns
    
    # Are the values non-negative for ghi, dni, dhi and wind speed
    assert (weather_data["ghi"] >= 0).all()
    assert (weather_data["dni"] >= 0).all()
    assert (weather_data["dhi"] >= 0).all()
    assert (weather_data["wind_speed"] >= 0).all()
    
    # Are the values output logical and realistic weather events
    assert weather_data["temp_air"].between(-30, 60).all()
    assert weather_data["wind_speed"].between(0, 60).all()
    assert weather_data["ghi"].max() <= 1500
    assert weather_data["dni"].max() <= 1500
    assert weather_data["dhi"].max() <= 1500
    assert weather_data["pressure"].between(70000, 110000).all()
    
    # Check date time index is increasing
    assert isinstance(weather_data.index, pd.DatetimeIndex)
    assert weather_data.index.is_monotonic_increasing
    
    print("Test Get Weather Data Passed")


def test_pv_profile_generator_tmy():
    EDP = EnergyDataProcessor(year=2026, lat=-36.67, lon=149.84)
    weather_data, _ = EDP.get_weather_data()

    pv_profile = pv_profile_generator_tmy(weather_data,-36.67,149.84)

    # Test PV Profiles are logical values and length
    assert len(pv_profile) == len(weather_data)
    assert np.isfinite(pv_profile).all()
    
    assert (pv_profile >= 0).all()
    assert (pv_profile > 0).any()
    assert (pv_profile == 0).any()
    
    # Maximum generation per kWp is realistic
    assert pv_profile.max() <= 1.2
    
    
def test_get_renewables_profiles():
    EDP = EnergyDataProcessor(year=2026, lat=-36.67, lon=149.84)
    weather_data, _ = EDP.get_weather_data()
    pv_profile = EDP.get_renewable_profiles(weather_data)

    assert isinstance(pv_profile, np.ndarray)
    assert len(pv_profile) == len(weather_data)
    assert np.isfinite(pv_profile).all()
    assert (pv_profile >= 0).all()
    
    
def test_zero_irradiance_gives_zero_pv():

    times = pd.date_range("2026-01-01", periods=24, freq="h")

    weather = pd.DataFrame({
        "temp_air": 20.0,
        "wind_speed": 2.0,
        "ghi": 0.0,
        "dni": 0.0,
        "dhi": 0.0,
        "pressure": 101325.0
    }, index=times)

    pv_profile = pv_profile_generator_tmy(weather, latitude=-36.67, longitude=149.84)

    np.testing.assert_allclose(pv_profile, 0, atol=1e-10)


def test_no_solar_input_gives_no_generation():

    EDP = EnergyDataProcessor(year=2026,lat=-36.67, lon=149.84)
    weather_data, _ = EDP.get_weather_data()

    pv_profile = pv_profile_generator_tmy(weather_data, EDP.lat, EDP.lon)

    zero_solar = ((weather_data["ghi"] == 0) & (weather_data["dni"] == 0) & (weather_data["dhi"] == 0))

    assert np.allclose(pv_profile[zero_solar], 0, atol=1e-10)
    

def test_create_pv_profile_csv(tmp_path):

    EDP = EnergyDataProcessor(year=2026, lat=-36.67, lon=149.84)
    output_file = tmp_path / "test_pv_profile.csv"
    pv_output = EDP.create_pv_profile_csv(output_file)

    # Correct columns
    assert list(pv_output.columns) == ["Time", "PV_Profile"]
    # Correct length
    assert len(pv_output) > 0
    # Logical PV values
    assert (pv_output["PV_Profile"] >= 0).all()
    assert np.isfinite(pv_output["PV_Profile"]).all()
    # File was actually created
    assert output_file.exists()
    

def test_build_time_index():

    EDP = EnergyDataProcessor(year=2026, lat=-36.67, lon=149.84)
    df = EDP.build_time_index(n_timesteps=365 * 24,dt_hours=1)

    assert len(df) == 8760
    assert isinstance(df.index, pd.DatetimeIndex)
    # Correct year
    assert df.index[0] == pd.Timestamp("2026-01-01 00:00:00")
    # Hourly spacing
    assert (df.index[1] - df.index[0]) == pd.Timedelta(hours=1)


def test_grid_price_profile():
    EDP = EnergyDataProcessor(year=2026, lat=-36.67, lon=149.84)
    times = pd.date_range("2026-01-01",periods=24,freq="h")
    df = pd.DataFrame(index=times)
    prices = EDP.get_grid_price_profile(df,onpeak_price=0.40,offpeak_price=0.20)

    assert len(prices) == 24
    # Before on-peak
    assert prices[14] == 0.20
    # On-peak
    assert prices[15] == 0.40
    assert prices[20] == 0.40
    # 21:00 returns to off-peak
    assert prices[21] == 0.20


def test_feed_in_tariff_profile():

    EDP = EnergyDataProcessor(year=2026, lat=-36.67, lon=149.84)
    times = pd.date_range("2026-01-01",periods=24,freq="h")
    df = pd.DataFrame(index=times)
    tariff = EDP.get_feed_in_tariff_profile(df,fit_on_peak=0.15,fit_off_peak=0.10,fit_flat=0.05)

    assert tariff[9] == 0.05
    assert tariff[10] == 0.15
    assert tariff[15] == 0.15
    assert tariff[16] == 0.10
    assert tariff[20] == 0.10
    assert tariff[21] == 0.05


def test_network_tariff_classification():

    EDP = EnergyDataProcessor(year=2026, lat=-36.67, lon=149.84)
    times = pd.date_range("2026-01-05",periods=24,freq="h")
    df = pd.DataFrame(index=times)

    result, month_limits, access_cost = (EDP.get_network_tariff_data(df))

    # Every timestep must be only one of the categories
    assert (result["peak"]+ result["shoulder"]+ result["offpeak"]== 1).all()
    # No negative network rates
    assert (result["network_energy_rate"] >= 0).all()
    # Access cost should be positive
    assert access_cost > 0
    
    # 2026-01-05 is a Monday
    # Monday 10:00 should be peak
    assert result.loc["2026-01-05 10:00", "peak"] == 1
    # Monday 07:00 should be shoulder
    assert result.loc["2026-01-05 07:00", "shoulder"] == 1
    # Midnight should be off-peak
    assert result.loc["2026-01-05 00:00", "offpeak"] == 1
    
    # 2026-01-03 is a Saturday
    times = pd.date_range("2026-01-03",periods=24,freq="h")
    df = pd.DataFrame(index=times)
    result, _, _ = EDP.get_network_tariff_data(df)
    # Peak and Shoulder don't apply on weekends
    assert (result["peak"] == 0).all()
    assert (result["shoulder"] == 0).all()
    assert (result["offpeak"] == 1).all()


