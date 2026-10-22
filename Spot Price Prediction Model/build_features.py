
"""
Build the training table: half-hourly price/demand joined with that month's
generation-mix shares, plus calendar and residual-demand features.
"""
import numpy as np
import pandas as pd

from load_price_demand import load_price_demand_history
from fuelmix import load_monthly_fuelmix_shares

# Approximate NSW public holidays are not modelled per-year here (would need
# a holidays library / calendar file) - flagged as a known simplification.
FEATURE_COLS = None  # set at bottom after build_training_table defines them


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    # Cyclical encodings so the model sees hour 23 and hour 0 as adjacent
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # NEM's southern-hemisphere summer (Dec-Feb) and winter (Jun-Aug) peaks
    df["is_summer"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_winter"] = df["month"].isin([6, 7, 8]).astype(int)
    return df


def build_training_table() -> pd.DataFrame:
    price_demand = load_price_demand_history()
    fuelmix = load_monthly_fuelmix_shares()

    df = price_demand.copy()
    df["year_month"] = df["datetime"].dt.to_period("M")
    df = df.merge(fuelmix, on="year_month", how="left")

    missing_fuelmix = df["coal_share"].isna().sum()
    if missing_fuelmix:
        print(f"[build_features] Warning: {missing_fuelmix} rows have no "
              f"matching fuel-mix month (outside fuel-mix file coverage) — "
              f"dropping them.")
        df = df.dropna(subset=["coal_share"])

    df = add_calendar_features(df)

    # Residual-demand proxy: demand not covered by variable renewables,
    # using that month's wind+solar share as a coarse stand-in for the
    # (unavailable at half-hourly resolution) actual wind/solar MW.
    # This is the single most important price-driving feature we can build
    # from monthly-resolution fuel data
    df["renewable_share"] = df["wind_share"] + df["solar_utility_share"] + df["solar_rooftop_share"]
    df["residual_demand_proxy"] = df["TOTALDEMAND"] * (1 - df["renewable_share"])

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


FEATURE_COLS = [
    "TOTALDEMAND", "residual_demand_proxy",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_weekend", "is_summer", "is_winter",
    "coal_share", "gas_share", "hydro_share", "wind_share",
    "solar_utility_share", "solar_rooftop_share", "battery_share",
    "monthly_temp_max", "monthly_temp_mean",
]
TARGET_COL = "RRP"


if __name__ == "__main__":
    df = build_training_table()
    print(df.shape)
    print(df[FEATURE_COLS + [TARGET_COL]].describe().T)
