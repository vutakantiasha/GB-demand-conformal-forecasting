from __future__ import annotations

from pathlib import Path

import pandas as pd

from gb_demand_forecasting.conformal.methods import adaptive_cqr, split_conformal_residual, static_cqr
from gb_demand_forecasting.utils.config import load_yaml
from gb_demand_forecasting.utils.io import ensure_dir


def calibrate_predictions(
    prediction_files: list[Path],
    conformal_cfg_path: Path,
    out_dir: Path,
) -> list[Path]:
    cfg = load_yaml(conformal_cfg_path)
    coverage = float(cfg["coverage"])
    horizon_aware = bool(cfg.get("horizon_aware", True))
    min_scores = int(cfg.get("min_scores_per_horizon", 20))
    adaptive_cfg = cfg["adaptive"]
    ensure_dir(out_dir)
    written: list[Path] = []
    for path in prediction_files:
        pred = pd.read_csv(path, parse_dates=["settlement_date"])
        model = str(pred["model"].iloc[0])
        calibration = pred[pred["split"] == "calibration"].copy()
        test = pred[pred["split"] == "test"].copy()
        if model == "seasonal_naive":
            calibrated = split_conformal_residual(
                calibration,
                test,
                coverage=coverage,
                horizon_aware=horizon_aware,
                min_scores_per_horizon=min_scores,
            )
            calibrated["model_variant"] = "seasonal_naive_split_conformal"
            out = out_dir / "seasonal_naive_split_conformal.csv"
            calibrated.to_csv(out, index=False)
            written.append(out)
            continue
        raw = test.copy()
        raw["calibration_method"] = "raw_quantile"
        raw["model_variant"] = f"{model}_raw"
        raw_path = out_dir / f"{model}_raw.csv"
        raw.to_csv(raw_path, index=False)
        written.append(raw_path)

        static = static_cqr(
            calibration,
            test,
            coverage=coverage,
            horizon_aware=horizon_aware,
            min_scores_per_horizon=min_scores,
        )
        static["model_variant"] = f"{model}_static_cqr"
        static_path = out_dir / f"{model}_static_cqr.csv"
        static.to_csv(static_path, index=False)
        written.append(static_path)

        adaptive = adaptive_cqr(
            calibration,
            test,
            coverage=coverage,
            gamma=float(adaptive_cfg["gamma"]),
            rolling_window=int(adaptive_cfg["rolling_window"]),
            alpha_min=float(adaptive_cfg["alpha_min"]),
            alpha_max=float(adaptive_cfg["alpha_max"]),
            horizon_aware=horizon_aware,
            min_scores_per_horizon=min_scores,
        )
        adaptive["model_variant"] = f"{model}_adaptive_cqr"
        adaptive_path = out_dir / f"{model}_adaptive_cqr.csv"
        adaptive.to_csv(adaptive_path, index=False)
        written.append(adaptive_path)
    return written
