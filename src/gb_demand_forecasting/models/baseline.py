from __future__ import annotations

import pandas as pd


def seasonal_naive_predictions(df: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
    lag_col = f"demand_lag_{lag_days}d_mw"
    if lag_col not in df.columns:
        raise KeyError(f"Missing seasonal naive feature: {lag_col}")
    out = df[
        ["settlement_date", "settlement_period", "split", "national_demand_mw"]
    ].copy()
    out["model"] = "seasonal_naive"
    out["q05"] = pd.NA
    out["q50"] = df[lag_col].values
    out["q95"] = pd.NA
    return out
