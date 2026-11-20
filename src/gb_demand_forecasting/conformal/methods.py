from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def conformity_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum(lower - y, y - upper)


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    x = np.asarray(scores, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    level = np.ceil((len(x) + 1) * (1 - alpha)) / len(x)
    level = float(np.clip(level, 0.0, 1.0))
    return float(np.quantile(x, level, method="higher"))


def static_cqr(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    coverage: float = 0.90,
    horizon_aware: bool = True,
    min_scores_per_horizon: int = 20,
) -> pd.DataFrame:
    alpha = 1.0 - coverage
    cal = calibration.dropna(subset=["national_demand_mw", "q05", "q95"]).copy()
    cal["score"] = conformity_scores(
        cal["national_demand_mw"].to_numpy(float),
        cal["q05"].to_numpy(float),
        cal["q95"].to_numpy(float),
    )
    pooled_q = finite_sample_quantile(cal["score"].to_numpy(float), alpha)
    horizon_q: dict[int, float] = {}
    if horizon_aware:
        for sp, group in cal.groupby("settlement_period"):
            horizon_q[int(sp)] = (
                finite_sample_quantile(group["score"].to_numpy(float), alpha)
                if len(group) >= min_scores_per_horizon
                else pooled_q
            )
    out = test.copy()
    corrections = out["settlement_period"].map(horizon_q).fillna(pooled_q).astype(float)
    out["q05"] = pd.to_numeric(out["q05"], errors="coerce") - corrections
    out["q95"] = pd.to_numeric(out["q95"], errors="coerce") + corrections
    out["conformal_correction_mw"] = corrections
    out["calibration_method"] = "static_cqr"
    return out


@dataclass
class AdaptiveState:
    alpha: float
    scores: list[float]


def adaptive_cqr(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    coverage: float = 0.90,
    gamma: float = 0.02,
    rolling_window: int = 180,
    alpha_min: float = 0.01,
    alpha_max: float = 0.30,
    horizon_aware: bool = True,
    min_scores_per_horizon: int = 20,
) -> pd.DataFrame:
    base_alpha = 1.0 - coverage
    cal = calibration.dropna(subset=["national_demand_mw", "q05", "q95"]).copy()
    cal["score"] = conformity_scores(
        cal["national_demand_mw"].to_numpy(float),
        cal["q05"].to_numpy(float),
        cal["q95"].to_numpy(float),
    )
    pooled = cal["score"].astype(float).tolist()
    states: dict[int, AdaptiveState] = {}
    if horizon_aware:
        for sp in sorted(test["settlement_period"].dropna().astype(int).unique()):
            vals = cal.loc[cal["settlement_period"] == sp, "score"].astype(float).tolist()
            if len(vals) < min_scores_per_horizon:
                vals = pooled.copy()
            states[sp] = AdaptiveState(alpha=base_alpha, scores=vals[-rolling_window:])
    else:
        states[0] = AdaptiveState(alpha=base_alpha, scores=pooled[-rolling_window:])

    out = test.sort_values(["settlement_date", "settlement_period"]).copy()
    lows: list[float] = []
    highs: list[float] = []
    corrections: list[float] = []
    alphas: list[float] = []
    misses: list[int] = []
    for row in out.itertuples(index=False):
        key = int(row.settlement_period) if horizon_aware else 0
        state = states[key]
        q = finite_sample_quantile(np.asarray(state.scores, dtype=float), state.alpha)
        lower = float(row.q05) - q
        upper = float(row.q95) + q
        y = float(row.national_demand_mw)
        miss = int(y < lower or y > upper)
        # Gibbs & Candès-style adaptive miscoverage update.
        state.alpha = float(
            np.clip(state.alpha + gamma * (base_alpha - miss), alpha_min, alpha_max)
        )
        score = float(max(float(row.q05) - y, y - float(row.q95)))
        state.scores.append(score)
        if len(state.scores) > rolling_window:
            state.scores = state.scores[-rolling_window:]
        lows.append(lower)
        highs.append(upper)
        corrections.append(q)
        alphas.append(state.alpha)
        misses.append(miss)
    out["q05"] = lows
    out["q95"] = highs
    out["conformal_correction_mw"] = corrections
    out["adaptive_alpha_after_update"] = alphas
    out["prequential_miss"] = misses
    out["calibration_method"] = "adaptive_cqr"
    return out.reset_index(drop=True)


def split_conformal_residual(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    coverage: float = 0.90,
    horizon_aware: bool = True,
    min_scores_per_horizon: int = 20,
) -> pd.DataFrame:
    alpha = 1.0 - coverage
    cal = calibration.dropna(subset=["national_demand_mw", "q50"]).copy()
    cal["score"] = (cal["national_demand_mw"] - cal["q50"]).abs()
    pooled_q = finite_sample_quantile(cal["score"].to_numpy(float), alpha)
    horizon_q: dict[int, float] = {}
    if horizon_aware:
        for sp, group in cal.groupby("settlement_period"):
            horizon_q[int(sp)] = (
                finite_sample_quantile(group["score"].to_numpy(float), alpha)
                if len(group) >= min_scores_per_horizon
                else pooled_q
            )
    out = test.copy()
    q = out["settlement_period"].map(horizon_q).fillna(pooled_q).astype(float)
    out["q05"] = pd.to_numeric(out["q50"], errors="coerce") - q
    out["q95"] = pd.to_numeric(out["q50"], errors="coerce") + q
    out["conformal_correction_mw"] = q
    out["calibration_method"] = "split_conformal_residual"
    return out
