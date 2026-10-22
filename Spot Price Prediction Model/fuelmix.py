"""
Turn the monthly OpenElectricity/OpenNEM fuel-mix export (GWh per fuel type
per month, plus monthly volume-weighted price and temperature) into a
per-month "generation mix share" table that can be broadcast onto every
half-hour within that month.
"""
import pandas as pd
from config import FUELMIX_MONTHLY_FILE, FUELMIX_TECH_MAP


def load_monthly_fuelmix_shares() -> pd.DataFrame:
    df = pd.read_csv(FUELMIX_MONTHLY_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    tech_cols = list(FUELMIX_TECH_MAP.keys())
    missing = [c for c in tech_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Expected fuel-mix columns not found: {missing}")

    # Roll raw fuel columns up into technology buckets (e.g. CCGT+OCGT+distillate -> gas)
    tech_gwh = pd.DataFrame({"year_month": df["year_month"]})
    for tech in set(FUELMIX_TECH_MAP.values()):
        cols = [c for c, t in FUELMIX_TECH_MAP.items() if t == tech]
        tech_gwh[f"{tech}_gwh"] = df[cols].sum(axis=1)

    gwh_cols = [c for c in tech_gwh.columns if c.endswith("_gwh")]
    tech_gwh["total_gwh"] = tech_gwh[gwh_cols].sum(axis=1)

    # Negative Pumps/exports aren't in FUELMIX_TECH_MAP so total_gwh here is
    # NSW generation only (imports/exports/battery-charging are handled as
    # separate demand-side effects, not generation share).
    for tech in set(FUELMIX_TECH_MAP.values()):
        tech_gwh[f"{tech}_share"] = (tech_gwh[f"{tech}_gwh"] / tech_gwh["total_gwh"]).clip(lower=0)

    share_cols = [c for c in tech_gwh.columns if c.endswith("_share")]
    out = tech_gwh[["year_month"] + share_cols].copy()

    # Bring across monthly price/temperature context too — useful slow-moving features
    out["monthly_vwap"] = df["Volume Weighted Price - AUD/MWh"]
    out["monthly_temp_max"] = df["Temperature Max - C"]
    out["monthly_temp_mean"] = df["Temperature Mean - C"]

    return out


if __name__ == "__main__":
    shares = load_monthly_fuelmix_shares()
    print(shares.shape)
    print(shares.tail(10))
    # share columns should sum to ~1 (allow for "other"/rounding)
    share_cols = [c for c in shares.columns if c.endswith("_share")]
    print("\nShare-sum sanity check (should be close to 1.0):")
    print(shares[share_cols].sum(axis=1).describe())
