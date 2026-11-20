from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    mask = np.isfinite(y) & np.isfinite(q_pred)
    if not mask.any():
        return float("nan")
    err = y[mask] - q_pred[mask]
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    mask = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if not mask.any():
        return float("nan")
    yv, lo, hi = y[mask], lower[mask], upper[mask]
    width = hi - lo
    penalty_low = (2.0 / alpha) * (lo - yv) * (yv < lo)
    penalty_high = (2.0 / alpha) * (yv - hi) * (yv > hi)
    return float(np.mean(width + penalty_low + penalty_high))


def compute_metrics(df: pd.DataFrame, coverage: float = 0.90) -> dict[str, float | int | str]:
    y = pd.to_numeric(df["national_demand_mw"], errors="coerce").to_numpy(float)
    point = pd.to_numeric(df["q50"], errors="coerce").to_numpy(float)
    point_mask = np.isfinite(y) & np.isfinite(point)
    result: dict[str, float | int | str] = {"n": int(point_mask.sum())}
    if point_mask.any():
        err = y[point_mask] - point[point_mask]
        result["mae_mw"] = float(np.mean(np.abs(err)))
        result["rmse_mw"] = float(np.sqrt(np.mean(err**2)))
        result["bias_mw"] = float(np.mean(point[point_mask] - y[point_mask]))
    else:
        result.update({"mae_mw": np.nan, "rmse_mw": np.nan, "bias_mw": np.nan})
    if {"q05", "q95"}.issubset(df.columns):
        lower = pd.to_numeric(df["q05"], errors="coerce").to_numpy(float)
        upper = pd.to_numeric(df["q95"], errors="coerce").to_numpy(float)
        mask = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
        if mask.any():
            inside = (y[mask] >= lower[mask]) & (y[mask] <= upper[mask])
            result["picp"] = float(np.mean(inside))
            result["coverage_error"] = float(np.mean(inside) - coverage)
            result["mpiw_mw"] = float(np.mean(upper[mask] - lower[mask]))
            result["interval_score"] = interval_score(y, lower, upper, 1.0 - coverage)
            result["pinball_q05"] = pinball_loss(y, lower, 0.05)
            result["pinball_q50"] = pinball_loss(y, point, 0.50)
            result["pinball_q95"] = pinball_loss(y, upper, 0.95)
        else:
            result.update(
                {
                    "picp": np.nan,
                    "coverage_error": np.nan,
                    "mpiw_mw": np.nan,
                    "interval_score": np.nan,
                    "pinball_q05": np.nan,
                    "pinball_q50": pinball_loss(y, point, 0.50),
                    "pinball_q95": np.nan,
                }
            )
    return result


def settlement_band(sp: pd.Series) -> pd.Series:
    return pd.cut(
        sp,
        bins=[0, 12, 24, 36, 48],
        labels=["00:00-06:00", "06:00-12:00", "12:00-18:00", "18:00-24:00"],
    )
