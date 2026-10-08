"""
Train quantile-regression LightGBM models (P10 / P50 / P90) to predict NSW1
spot price (RRP) from demand and generation-mix features.

Spot price is heavy-tailed and spike-prone, so a single point forecast is
misleading - we train one model per quantile and report a band, not a
single number.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from build_features import build_training_table, FEATURE_COLS, TARGET_COL
from config import QUANTILES

MODEL_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split chronologically, not randomly - random k-fold would leak
    future information into training for a time series."""
    df = df.sort_values("datetime")
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_date = df.iloc[cutoff_idx]["datetime"]
    train = df[df["datetime"] < cutoff_date]
    test = df[df["datetime"] >= cutoff_date]
    return train, test


def train_quantile_models(train_df: pd.DataFrame) -> dict:
    models = {}
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    for q in QUANTILES:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, verbose=-1, **MODEL_PARAMS)
        model.fit(X_train, y_train)
        models[q] = model
    return models


def evaluate(models: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].values

    preds = {q: models[q].predict(X_test) for q in QUANTILES}

    p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
    mae_median = mean_absolute_error(y_test, p50)
    coverage = np.mean((y_test >= p10) & (y_test <= p90))  # should be ~80%

    print(f"Test rows: {len(y_test)}")
    print(f"MAE of median (P50) forecast: ${mae_median:,.2f}/MWh")
    print(f"P10-P90 empirical coverage: {coverage:.1%} (target ~80%)")

    result = test_df[["datetime", "TOTALDEMAND", "RRP"]].copy()
    result["pred_p10"] = p10
    result["pred_p50"] = p50
    result["pred_p90"] = p90
    return result


if __name__ == "__main__":
    df = build_training_table()
    train_df, test_df = time_based_split(df, test_frac=0.2)
    print(f"Train: {len(train_df)} rows ({train_df.datetime.min()} -> {train_df.datetime.max()})")
    print(f"Test:  {len(test_df)} rows ({test_df.datetime.min()} -> {test_df.datetime.max()})")

    models = train_quantile_models(train_df)
    results = evaluate(models, test_df)

    print("\nFeature importance (P50 model):")
    imp = pd.Series(models[0.5].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(imp)
