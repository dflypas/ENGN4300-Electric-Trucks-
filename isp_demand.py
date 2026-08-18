"""
Reshape AEMO's ISP demand trace (wide: one row per day, 48 half-hour columns)
into the same long half-hourly format as the historical price/demand files.
"""
import pandas as pd
from config import ISP_DEMAND_FILE


def load_isp_demand_long(year_start: int = None, year_end: int = None) -> pd.DataFrame:
    df = pd.read_csv(ISP_DEMAND_FILE)

    hh_cols = [c for c in df.columns if c not in ("Year", "Month", "Day")]
    long_df = df.melt(id_vars=["Year", "Month", "Day"], value_vars=hh_cols,
                       var_name="period", value_name="demand_mw")
    long_df["period"] = long_df["period"].astype(int)

    # Period 1 = 00:30, period 48 = 24:00 (next-day 00:00), standard AEMO convention
    minutes = long_df["period"] * 30
    base_date = pd.to_datetime(dict(year=long_df.Year, month=long_df.Month, day=long_df.Day))
    long_df["datetime"] = base_date + pd.to_timedelta(minutes, unit="m")

    long_df = long_df[["datetime", "demand_mw"]].sort_values("datetime").reset_index(drop=True)

    if year_start is not None:
        long_df = long_df[long_df["datetime"].dt.year >= year_start]
    if year_end is not None:
        long_df = long_df[long_df["datetime"].dt.year <= year_end]

    return long_df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_isp_demand_long(year_start=2026, year_end=2036)
    print(df.shape)
    print(df.head())
    print(df.tail())
    print("\nAnnual peak demand (MW), Step Change POE50:")
    print(df.groupby(df.datetime.dt.year)["demand_mw"].max().round(0))
