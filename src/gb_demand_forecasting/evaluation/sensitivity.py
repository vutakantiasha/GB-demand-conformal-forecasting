from __future__ import annotations

from pathlib import Path

import pandas as pd

from gb_demand_forecasting.conformal.methods import adaptive_cqr, static_cqr
from gb_demand_forecasting.evaluation.metrics import compute_metrics


def conformal_sensitivity(
    prediction_files: list[Path],
    conformal_cfg: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    levels = [float(x) for x in conformal_cfg.get("sensitivity_levels", [0.8, 0.9, 0.95])]
    adaptive = conformal_cfg.get("adaptive", {})
    sens = conformal_cfg.get("adaptive_sensitivity", {})
    gammas = [float(x) for x in sens.get("gamma", [adaptive.get("gamma", 0.02)])]
    windows = [int(x) for x in sens.get("rolling_window", [adaptive.get("rolling_window", 180)])]
    for path in prediction_files:
        pred = pd.read_csv(path, parse_dates=["settlement_date"])
        model = str(pred["model"].iloc[0])
        if model == "seasonal_naive" or pred[["q05", "q95"]].isna().all().all():
            continue
        cal = pred[pred["split"] == "calibration"].copy()
        test = pred[pred["split"] == "test"].copy()
        for level in levels:
            static = static_cqr(
                cal,
                test,
                coverage=level,
                horizon_aware=bool(conformal_cfg.get("horizon_aware", True)),
                min_scores_per_horizon=int(conformal_cfg.get("min_scores_per_horizon", 20)),
            )
            rows.append(
                {
                    "model": model,
                    "method": "static_cqr",
                    "coverage_target": level,
                    "gamma": None,
                    "rolling_window": None,
                    **compute_metrics(static, coverage=level),
                }
            )
            for gamma in gammas:
                for window in windows:
                    out = adaptive_cqr(
                        cal,
                        test,
                        coverage=level,
                        gamma=gamma,
                        rolling_window=window,
                        alpha_min=float(adaptive.get("alpha_min", 0.01)),
                        alpha_max=float(adaptive.get("alpha_max", 0.30)),
                        horizon_aware=bool(conformal_cfg.get("horizon_aware", True)),
                        min_scores_per_horizon=int(conformal_cfg.get("min_scores_per_horizon", 20)),
                    )
                    rows.append(
                        {
                            "model": model,
                            "method": "adaptive_cqr",
                            "coverage_target": level,
                            "gamma": gamma,
                            "rolling_window": window,
                            **compute_metrics(out, coverage=level),
                        }
                    )
    return pd.DataFrame(rows)
