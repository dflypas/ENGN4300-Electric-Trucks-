"""
Load AEMO PRICE_AND_DEMAND_<YYYYMM>_<REGION>.csv files (one file per month,
the standard public NEM format) and concatenate into one clean half-hourly
(or 5-minute, for post-Oct-2021 files) time series.
"""
import glob
import pandas as pd
from config import PRICE_DEMAND_GLOB, REGION


def load_price_demand_history() -> pd.DataFrame:
    files = sorted(glob.glob(PRICE_DEMAND_GLOB))
    if not files:
        raise FileNotFoundError(f"No price/demand files found matching {PRICE_DEMAND_GLOB}")

    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["SETTLEMENTDATE"])
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Dedup: files can overlap by one boundary row (e.g. month-end 00:00
    # appears as both the last row of one file and isn't in the next) —
    # keep first occurrence per timestamp.
    df = df.drop_duplicates(subset=["REGION", "SETTLEMENTDATE"], keep="first")
    df = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)

    df = df[df["REGION"] == REGION].copy()

    # Sanity checks worth surfacing rather than silently trusting the data
    gaps = df["SETTLEMENTDATE"].diff().dropna()
    expected = pd.Timedelta(minutes=30)
    n_gaps = (gaps > expected).sum()
    if n_gaps:
        print(f"[load_price_demand] Warning: {n_gaps} time gaps larger than "
              f"30 min detected — likely missing months. Fill coverage before "
              f"training on the full history.")

    return df[["SETTLEMENTDATE", "TOTALDEMAND", "RRP"]].rename(
        columns={"SETTLEMENTDATE": "datetime"}
    )


if __name__ == "__main__":
    df = load_price_demand_history()
    print(df.shape)
    print(df["datetime"].min(), "->", df["datetime"].max())
    print(df.describe())
