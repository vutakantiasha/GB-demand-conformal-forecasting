from __future__ import annotations

import numpy as np
import pandas as pd


def daily_block_bootstrap_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str = "mae",
    replicates: int = 1000,
    seed: int = 2026,
) -> dict[str, float | int | str]:
    keys = ["settlement_date", "settlement_period"]
    a = left[keys + ["national_demand_mw", "q50"]].rename(columns={"q50": "left"})
    b = right[keys + ["q50"]].rename(columns={"q50": "right"})
    merged = a.merge(b, on=keys, how="inner").dropna()
    if merged.empty:
        return {"estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_days": 0}
    merged["settlement_date"] = pd.to_datetime(merged["settlement_date"]).dt.date
    days = np.array(sorted(merged["settlement_date"].unique()), dtype=object)
    by_day = {d: merged[merged["settlement_date"] == d] for d in days}

    def score(frame: pd.DataFrame, col: str) -> float:
        e = frame["national_demand_mw"].to_numpy(float) - frame[col].to_numpy(float)
        if metric == "rmse":
            return float(np.sqrt(np.mean(e**2)))
        return float(np.mean(np.abs(e)))

    estimate = score(merged, "left") - score(merged, "right")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for i in range(replicates):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        frame = pd.concat([by_day[d] for d in sampled_days], ignore_index=True)
        samples[i] = score(frame, "left") - score(frame, "right")
    return {
        "metric": metric,
        "estimate": float(estimate),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "n_days": int(len(days)),
        "replicates": int(replicates),
    }
