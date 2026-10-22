"""Simple verification tests V39-V43 for the energy data processor."""

import numpy as np
import pandas as pd
import pytest

from energy_data_processor import EnergyDataProcessor, pv_profile_generator_tmy, NETWORK_TARIFF


def make_weather():
    """Small weather dataset with daylight and night-time hours."""
    times = pd.date_range("2026-01-15", periods=48, freq="h")
    daylight = (times.hour >= 7) & (times.hour <= 18)
    return pd.DataFrame({
        "temp_air": 20.0,
        "wind_speed": 2.0,
        "ghi": np.where(daylight, 700.0, 0.0),
        "dni": np.where(daylight, 600.0, 0.0),
        "dhi": np.where(daylight, 100.0, 0.0),
        "pressure": 101325.0,
    }, index=times)


def test_v39_weather_data():
    # Check the actual weather data returned by PVGIS.
    p = EnergyDataProcessor(2026, -36.67, 149.84)
    data, elevation = p.get_weather_data()

    required = ["temp_air", "wind_speed", "ghi", "dni", "dhi", "pressure"]
    assert all(column in data.columns for column in required)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.is_monotonic_increasing
    assert data[required].notna().all().all()
    assert (data[["wind_speed", "ghi", "dni", "dhi"]] >= 0).all().all()
    assert data["temp_air"].between(-50, 60).all()
    assert data["pressure"].between(50000, 110000).all()
    assert np.isfinite(elevation)


def test_v40_pv_profile():
    weather = make_weather()
    profile = np.asarray(pv_profile_generator_tmy(weather, -36.67, 149.84))
    assert len(profile) == len(weather)
    assert np.isfinite(profile).all()
    assert (profile >= 0).all()
    assert (profile == 0).any()
    assert (profile > 0).any()
    assert profile.max() <= 1.5  # Simple sanity bound for normalised output


def test_v41_zero_irradiance():
    weather = make_weather().iloc[:24].copy()
    weather[["ghi", "dni", "dhi"]] = 0
    profile = np.asarray(pv_profile_generator_tmy(weather, -36.67, 149.84))
    assert np.allclose(profile, 0, atol=1e-6)

    # Also check night-time hours in the normal weather dataset.
    weather = make_weather()
    profile = np.asarray(pv_profile_generator_tmy(weather, -36.67, 149.84))
    night = (weather[["ghi", "dni", "dhi"]] == 0).all(axis=1).to_numpy()
    assert np.allclose(profile[night], 0, atol=1e-6)


def test_v42_create_pv_csv(tmp_path):
    # Run the complete weather-to-PV CSV export.
    p = EnergyDataProcessor(2026, -36.67, 149.84)
    output_file = tmp_path / "pv_profile_output.csv"
    p.create_pv_profile_csv(output_file)

    assert output_file.exists()
    result = pd.read_csv(output_file)
    assert len(result) > 0
    assert list(result.columns) == ["Time", "PV_Profile"]
    assert np.isfinite(result["PV_Profile"]).all()
    assert (result["PV_Profile"] >= 0).all()


def test_v43_network_tariff():
    p = EnergyDataProcessor(2026, -36.67, 149.84)
    times = pd.to_datetime([
        "2026-10-19 06:00",  # Monday off-peak
        "2026-10-19 08:00",  # Monday shoulder
        "2026-10-19 12:00",  # Monday peak
        "2026-10-19 16:00",  # Monday shoulder
        "2026-10-19 18:00",  # Monday peak
        "2026-10-19 21:00",  # Monday shoulder
        "2026-10-24 12:00",  # Saturday off-peak
    ])
    result, month_limits, access_cost = p.get_network_tariff_data(pd.DataFrame(index=times))
    assert result["peak"].tolist() == [0, 0, 1, 0, 1, 0, 0]
    assert result["shoulder"].tolist() == [0, 1, 0, 1, 0, 1, 0]
    assert result["offpeak"].tolist() == [1, 0, 0, 0, 0, 0, 1]
    assert (result[["peak", "shoulder", "offpeak"]].sum(axis=1) == 1).all()
    expected_rates = [
        NETWORK_TARIFF["energy_rate_offpeak"],
        NETWORK_TARIFF["energy_rate_shoulder"],
        NETWORK_TARIFF["energy_rate_peak"],
        NETWORK_TARIFF["energy_rate_shoulder"],
        NETWORK_TARIFF["energy_rate_peak"],
        NETWORK_TARIFF["energy_rate_shoulder"],
        NETWORK_TARIFF["energy_rate_offpeak"],
    ]
    assert result["network_energy_rate"].tolist() == pytest.approx(expected_rates)
    assert month_limits == {0: 6}
    assert access_cost == pytest.approx(NETWORK_TARIFF["network_access_charge"] * 7 / 24)

