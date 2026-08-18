import numpy as np
import pandas as pd

from build_features import add_calendar_features, FEATURE_COLS
from capacity_forward import build_forward_capacity_by_year, capacity_to_shares
from isp_demand import load_isp_demand_long
from fuelmix import load_monthly_fuelmix_shares
from train_model import build_training_table, train_quantile_models, time_based_split
from config import QUANTILES, MARKET_PRICE_FLOOR, market_price_cap


def build_temperature_climatology() -> pd.DataFrame:
    """Average historical monthly temps by calendar month, as a stand-in
    for future weather. This assumes climate statistics stay put over
    2026-2036, which is a real (if usually second-order for price)
    simplification - flagged here rather than buried."""
    hist = load_monthly_fuelmix_shares()
    hist["month"] = hist["year_month"].dt.month
    clim = hist.groupby("month")[["monthly_temp_max", "monthly_temp_mean"]].mean()
    return clim


def build_forward_feature_table(year_start=2026, year_end=2036) -> pd.DataFrame:
    demand = load_isp_demand_long(year_start=year_start, year_end=year_end)
    demand = demand.rename(columns={"demand_mw": "TOTALDEMAND"})

    cap_by_year = build_forward_capacity_by_year()
    shares_by_year = capacity_to_shares(cap_by_year)

    df = demand.copy()
    df["year"] = df["datetime"].dt.year
    # Capacity file's earliest column is the 2027 summer; use it for 2026 too
    # (no material fleet difference expected between the two).
    available_years = shares_by_year.index
    df["cap_year"] = df["year"].clip(lower=available_years.min(), upper=available_years.max())
    df = df.merge(shares_by_year, left_on="cap_year", right_index=True, how="left")

    # No rooftop-solar capacity in the AEMO generator register (behind-the-meter,
    # not centrally registered) - hold it at the last historically observed share
    # as a simplification, since it's a small, fairly stable slice.
    hist = load_monthly_fuelmix_shares()
    df["solar_rooftop_share"] = hist["solar_rooftop_share"].iloc[-12:].mean()

    df = add_calendar_features(df)

    clim = build_temperature_climatology()
    df = df.merge(clim, left_on="month", right_index=True, how="left")

    df["renewable_share"] = df["wind_share"] + df["solar_utility_share"] + df["solar_rooftop_share"]
    df["residual_demand_proxy"] = df["TOTALDEMAND"] * (1 - df["renewable_share"])

    return df


def run_forecast(models: dict, forward_df: pd.DataFrame) -> pd.DataFrame:
    X = forward_df[FEATURE_COLS]
    out = forward_df[["datetime", "year", "TOTALDEMAND"]].copy()
    caps = out["year"].map(market_price_cap).values
    for q in QUANTILES:
        pred = models[q].predict(X)
        out[f"pred_p{int(q*100)}"] = np.clip(pred, MARKET_PRICE_FLOOR, caps)
    return out


def summarize_by_year(forecast_df: pd.DataFrame) -> pd.DataFrame:
    summary = forecast_df.groupby("year").agg(
        avg_p10=("pred_p10", "mean"),
        avg_p50=("pred_p50", "mean"),
        avg_p90=("pred_p90", "mean"),
        max_p50=("pred_p50", "max"),
    ).round(1)
    return summary


if __name__ == "__main__":
    hist_df = build_training_table()
    # Use ALL history to fit the models used for the forward run (the
    # earlier time_based_split was for backtesting/validation only)
    models = train_quantile_models(hist_df)

    forward_df = build_forward_feature_table(2026, 2036)
    forecast = run_forecast(models, forward_df)

    print("Forward feature table sample:")
    print(forward_df[["datetime"] + FEATURE_COLS].head())

    print("\n10-year NSW1 spot price forecast summary ($/MWh):")
    print(summarize_by_year(forecast))

    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    forecast.to_csv("data/forecast_2026_2036.csv", index=False)
    print("\nSaved full half-hourly forecast to data/forecast_2026_2036.csv")


"""
Generate static PNG charts from the forecast output:
  1. Annual P10/P50/P90 price band, 2026-2036
  2. Average price by hour of day (duck curve / evening peak)
  3. Average price by month (seasonal pattern)

Run after forecast.py has produced data/forecast_2026_2036.csv.
"""
import matplotlib.pyplot as plt
import pandas as pd

FORECAST_FILE = "data/forecast_2026_2036.csv"
OUT_DIR = "data/charts"


def load_forecast() -> pd.DataFrame:
    df = pd.read_csv(FORECAST_FILE, parse_dates=["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    return df


def plot_annual(df: pd.DataFrame, out_dir: str):
    yearly = df.groupby("year")[["pred_p10", "pred_p50", "pred_p90"]].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(yearly.index, yearly["pred_p10"], yearly["pred_p90"],
                     alpha=0.2, color="#2a78d6", label="P10-P90 band")
    ax.plot(yearly.index, yearly["pred_p50"], color="#2a78d6", marker="o",
             linewidth=2, label="P50 (median)")
    ax.set_title("NSW1 spot price forecast, 2026-2036")
    ax.set_xlabel("Year")
    ax.set_ylabel("$/MWh")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/annual_forecast.png", dpi=150)
    plt.close(fig)


def plot_hourly(df: pd.DataFrame, out_dir: str):
    hourly = df.groupby("hour")[["pred_p10", "pred_p50", "pred_p90"]].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(hourly.index, hourly["pred_p10"], hourly["pred_p90"],
                     alpha=0.2, color="#2a78d6", label="P10-P90 band")
    ax.plot(hourly.index, hourly["pred_p50"], color="#2a78d6", linewidth=2, label="P50 (median)")
    ax.set_title("Average forecast price by hour of day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("$/MWh")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/hourly_pattern.png", dpi=150)
    plt.close(fig)


def plot_monthly(df: pd.DataFrame, out_dir: str):
    monthly = df.groupby("month")["pred_p50"].mean()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(month_labels, monthly.values, color="#2a78d6")
    ax.set_title("Average median forecast price by month")
    ax.set_ylabel("$/MWh")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/monthly_pattern.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path
    Path(OUT_DIR).mkdir(exist_ok=True)

    df = load_forecast()
    plot_annual(df, OUT_DIR)
    plot_hourly(df, OUT_DIR)
    plot_monthly(df, OUT_DIR)

    print(f"Saved 3 charts to {OUT_DIR}/")

    if __name__ == "__main__":
        from pathlib import Path
        out_path = Path(OUT_DIR).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        df = load_forecast()
        plot_annual(df, str(out_path))
        plot_hourly(df, str(out_path))
        plot_monthly(df, str(out_path))

        print(f"Saved 3 charts to: {out_path}")
        for f in out_path.glob("*.png"):
            print(f"  - {f}")