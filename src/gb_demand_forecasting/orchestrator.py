from __future__ import annotations

import json
import logging
import subprocess  # nosec B404 -- fixed argv; isolated worker requires subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from gb_demand_forecasting.conformal.pipeline import calibrate_predictions
from gb_demand_forecasting.data.ckan import acquire_neso
from gb_demand_forecasting.data.prepare import prepare_dataset
from gb_demand_forecasting.data.synthetic import acquire_synthetic
from gb_demand_forecasting.evaluation.evaluate import evaluate_all
from gb_demand_forecasting.evaluation.sensitivity import conformal_sensitivity
from gb_demand_forecasting.evaluation.interpretability import lightgbm_shap_analysis
from gb_demand_forecasting.features.matrix import primary_rows, split_frame
from gb_demand_forecasting.models.baseline import seasonal_naive_predictions
from gb_demand_forecasting.models.lightgbm_quantile import fit_quantile_models
from gb_demand_forecasting.reporting.plots import make_plots
from gb_demand_forecasting.reporting.report import build_results_report
from gb_demand_forecasting.utils.config import load_yaml, resolve_project_path
from gb_demand_forecasting.utils.io import ensure_dir, read_frame, write_json
from gb_demand_forecasting.utils.reproducibility import runtime_manifest, seed_everything

LOGGER = logging.getLogger(__name__)


def acquire(cfg: dict[str, Any], root: Path) -> Path:
    mode = cfg["source"]["mode"]
    if mode == "neso":
        return acquire_neso(cfg, root)
    if mode == "synthetic":
        return acquire_synthetic(cfg, root)
    raise ValueError(f"Unknown source.mode={mode}")


def prepare(cfg: dict[str, Any], root: Path) -> Path:
    return prepare_dataset(cfg, root)


def train(cfg: dict[str, Any], root: Path, master_path: Path | None = None) -> list[Path]:
    if master_path is None:
        master_path = root / cfg["paths"]["processed_dir"] / "master_dataset.csv.gz"
    master = read_frame(master_path)
    master["settlement_date"] = pd.to_datetime(master["settlement_date"]).dt.date
    master = primary_rows(
        master, exclude_dst=bool(cfg["forecast"].get("exclude_dst_transition_days_primary", True))
    )
    features = list(cfg["features"]["model_features"])
    missing = [f for f in features if f not in master.columns]
    if missing:
        raise KeyError(f"Configured features missing from master dataset: {missing}")

    artifacts_dir = ensure_dir(root / cfg["paths"]["artifacts_dir"])
    pred_dir = ensure_dir(artifacts_dir / "predictions")
    seed = int(cfg["project"]["random_seed"])
    prediction_files: list[Path] = []

    # Seasonal-naive transparent baseline.
    seasonal_input = pd.concat(
        [split_frame(master, "calibration"), split_frame(master, "test")], ignore_index=True
    )
    seasonal = seasonal_naive_predictions(
        seasonal_input, lag_days=int(cfg["models"]["seasonal_naive"]["lag_days"])
    )
    seasonal_path = pred_dir / "seasonal_naive.csv"
    seasonal.to_csv(seasonal_path, index=False)
    prediction_files.append(seasonal_path)

    # LightGBM quantile forecasting.
    lgb_cfg_path = resolve_project_path(root, cfg["models"]["lightgbm_config"])
    lgb_cfg = load_yaml(lgb_cfg_path)
    lgb_pred, lgb_meta = fit_quantile_models(
        split_frame(master, "training"),
        split_frame(master, "validation"),
        split_frame(master, "calibration"),
        split_frame(master, "test"),
        features,
        lgb_cfg,
        artifacts_dir,
        seed,
        model_name="lightgbm",
    )
    lgb_path = pred_dir / "lightgbm.csv"
    lgb_pred.to_csv(lgb_path, index=False)
    prediction_files.append(lgb_path)

    # E5 feature ablation: remove embedded renewable forecasts while keeping the same
    # model family, split discipline and tuning budget.
    ablation_features = [
        f for f in features if f not in {"embedded_wind_forecast_mw", "embedded_solar_forecast_mw"}
    ]
    ab_pred, ab_meta = fit_quantile_models(
        split_frame(master, "training"),
        split_frame(master, "validation"),
        split_frame(master, "calibration"),
        split_frame(master, "test"),
        ablation_features,
        lgb_cfg,
        artifacts_dir,
        seed,
        model_name="lightgbm_no_renewables",
    )
    ab_path = pred_dir / "lightgbm_no_renewables.csv"
    ab_pred.to_csv(ab_path, index=False)
    prediction_files.append(ab_path)

    # Research-grade Temporal Fusion Transformer implementation.
    # Run in a fresh interpreter so PyTorch and LightGBM never load competing OpenMP
    # runtimes in the same process on macOS Apple Silicon.
    tft_cfg_path = resolve_project_path(root, cfg["models"]["tft_config"])
    tft_path = pred_dir / "tft.csv"
    tft_meta_path = artifacts_dir / "models" / "tft" / "worker_meta.json"
    tft_meta_path.parent.mkdir(parents=True, exist_ok=True)
    worker_cmd = [
        sys.executable,
        "-m",
        "gb_demand_forecasting.models.tft_worker",
        "--master",
        str(master_path),
        "--features-json",
        json.dumps(features),
        "--tft-config",
        str(tft_cfg_path),
        "--artifacts-dir",
        str(artifacts_dir),
        "--output",
        str(tft_path),
        "--meta-output",
        str(tft_meta_path),
        "--seed",
        str(seed),
        "--device",
        str(cfg["runtime"].get("device", "auto")),
        "--num-workers",
        str(int(cfg["runtime"].get("num_workers", 0))),
    ]
    if bool(cfg["forecast"].get("exclude_dst_transition_days_primary", True)):
        worker_cmd.append("--exclude-dst")
    LOGGER.info("Launching TFT in isolated subprocess to avoid ML runtime collisions")
    subprocess.run(worker_cmd, cwd=root, check=True)  # noqa: S603
    if not tft_path.exists():
        raise RuntimeError("Isolated TFT worker exited without producing predictions")
    with tft_meta_path.open("r", encoding="utf-8") as handle:
        tft_meta = json.load(handle)
    prediction_files.append(tft_path)

    write_json(
        {"lightgbm": lgb_meta, "lightgbm_no_renewables": ab_meta, "tft": tft_meta},
        root / cfg["paths"]["reports_dir"] / "training_summary.json",
    )
    return prediction_files


def run_evaluation(
    cfg: dict[str, Any], root: Path, prediction_files: list[Path], master_path: Path
) -> dict[str, Path]:
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    conformal_cfg = resolve_project_path(root, cfg["conformal"]["config"])
    calibrated_dir = ensure_dir(artifacts_dir / "calibration")
    calibrated_files = calibrate_predictions(prediction_files, conformal_cfg, calibrated_dir)

    master = read_frame(master_path)
    master["settlement_date"] = pd.to_datetime(master["settlement_date"])
    master = primary_rows(
        master, exclude_dst=bool(cfg["forecast"].get("exclude_dst_transition_days_primary", True))
    )
    results_dir = ensure_dir(root / cfg["paths"]["results_dir"])
    outputs = evaluate_all(calibrated_files, prediction_files, master, cfg, results_dir)
    conformal_cfg_data = load_yaml(conformal_cfg)
    sensitivity = conformal_sensitivity(prediction_files, conformal_cfg_data)
    sensitivity_path = results_dir / "metrics" / "conformal_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)
    outputs["conformal_sensitivity"] = sensitivity_path
    plots = make_plots(
        calibrated_files,
        outputs["overall_metrics"],
        results_dir / "figures",
        rolling_days=int(cfg["evaluation"]["rolling_window_days"]),
    )
    outputs["figures_dir"] = results_dir / "figures"
    reports_dir = ensure_dir(root / cfg["paths"]["reports_dir"])
    report_path = build_results_report(
        cfg,
        outputs["overall_metrics"],
        outputs["bootstrap"],
        outputs["distribution_shift"],
        reports_dir / "data_quality.json",
        reports_dir / "results_evidence_pack.md",
    )
    outputs["results_report"] = report_path
    if bool(cfg["runtime"].get("run_shap", False)):
        shap_csv, shap_png = lightgbm_shap_analysis(
            master,
            artifacts_dir / "models" / "lightgbm" / "quantile_0.50.txt",
            artifacts_dir / "models" / "lightgbm" / "feature_schema.json",
            results_dir / "metrics" / "lightgbm_shap_importance.csv",
            results_dir / "figures" / "06_lightgbm_shap_summary.png",
            max_rows=int(cfg["runtime"].get("shap_max_rows", 2000)),
            seed=int(cfg["project"]["random_seed"]),
        )
        outputs["lightgbm_shap_importance"] = shap_csv
        outputs["lightgbm_shap_figure"] = shap_png
    LOGGER.info("Generated %d core figures", len(plots))
    return outputs


def full_run(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    seed_everything(int(cfg["project"]["random_seed"]))
    LOGGER.info("Starting full pipeline | source=%s", cfg["source"]["mode"])
    manifest_path = acquire(cfg, root)
    master_path = prepare(cfg, root)
    prediction_files = train(cfg, root, master_path)
    outputs = run_evaluation(cfg, root, prediction_files, master_path)
    reports_dir = ensure_dir(root / cfg["paths"]["reports_dir"])
    manifest = runtime_manifest(root, cfg)

    def _portable_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return str(path)

    manifest["source_manifest"] = _portable_path(manifest_path)
    manifest["master_dataset"] = _portable_path(master_path)
    manifest["prediction_files"] = [_portable_path(p) for p in prediction_files]
    manifest["outputs"] = {k: _portable_path(v) for k, v in outputs.items()}
    manifest_path_out = write_json(manifest, reports_dir / "reproducibility_manifest.json")
    LOGGER.info("Full pipeline completed")
    return {
        "source_manifest": manifest_path,
        "master_dataset": master_path,
        "predictions": prediction_files,
        "reproducibility_manifest": manifest_path_out,
        **outputs,
    }
