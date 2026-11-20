from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gb_demand_forecasting.evaluation.bootstrap import daily_block_bootstrap_difference
from gb_demand_forecasting.evaluation.metrics import compute_metrics, settlement_band
from gb_demand_forecasting.evaluation.shift import distribution_shift_report
from gb_demand_forecasting.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


def _load_variant(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["settlement_date"])
    if "model_variant" not in df.columns:
        df["model_variant"] = path.stem
    return df


def evaluate_all(
    calibrated_files: list[Path],
    point_files: list[Path],
    master: pd.DataFrame,
    cfg: dict,
    results_dir: Path,
) -> dict[str, Path]:
    metrics_dir = ensure_dir(results_dir / "metrics")
    error_dir = ensure_dir(results_dir / "error_analysis")
    bootstrap_dir = ensure_dir(results_dir / "bootstrap")
    coverage = float(cfg["evaluation"]["primary_interval_level"])

    variants: dict[str, pd.DataFrame] = {}
    for path in calibrated_files:
        df = _load_variant(path)
        variants[str(df["model_variant"].iloc[0])] = df

    # Add the external NESO point benchmark directly from the master table.
    neso = master[master["split"] == "test"][
        ["settlement_date", "settlement_period", "national_demand_mw", "neso_day_ahead_forecast_mw"]
    ].copy()
    neso = neso.dropna(subset=["neso_day_ahead_forecast_mw"])
    if not neso.empty:
        neso["q50"] = neso["neso_day_ahead_forecast_mw"]
        neso["q05"] = np.nan
        neso["q95"] = np.nan
        neso["model_variant"] = "neso_operational_external"
        variants["neso_operational_external"] = neso

    rows = []
    for name, frame in variants.items():
        metric = compute_metrics(frame, coverage=coverage)
        rows.append({"model_variant": name, **metric})
    metrics = pd.DataFrame(rows).sort_values("mae_mw")
    metrics_path = metrics_dir / "overall_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    slice_rows: list[dict] = []
    master_attrs = master[
        ["settlement_date", "settlement_period", "season", "is_weekend", "national_demand_mw"]
    ].copy()
    train_target = master.loc[master["split"] == "training", "national_demand_mw"].dropna()
    quantiles = train_target.quantile(np.linspace(0, 1, 11)).to_numpy(float)
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    for name, frame in variants.items():
        joined = frame.drop(columns=["national_demand_mw"], errors="ignore").merge(
            master_attrs, on=["settlement_date", "settlement_period"], how="left"
        )
        joined["settlement_band"] = settlement_band(joined["settlement_period"])
        joined["demand_decile"] = pd.cut(
            joined["national_demand_mw"], bins=np.unique(quantiles), labels=False, include_lowest=True
        )
        for dimension in ["season", "settlement_band", "is_weekend", "demand_decile"]:
            for value, group in joined.groupby(dimension, observed=True):
                slice_rows.append(
                    {
                        "model_variant": name,
                        "dimension": dimension,
                        "value": str(value),
                        **compute_metrics(group, coverage=coverage),
                    }
                )
    slices_path = metrics_dir / "slice_metrics.csv"
    pd.DataFrame(slice_rows).to_csv(slices_path, index=False)

    # Top-error-day evidence pack.
    error_rows: list[pd.DataFrame] = []
    top_n = int(cfg["evaluation"]["top_error_days"])
    for name, frame in variants.items():
        f = frame.copy()
        f["abs_error_mw"] = (f["national_demand_mw"] - f["q50"]).abs()
        daily = (
            f.groupby("settlement_date")["abs_error_mw"]
            .mean()
            .nlargest(top_n)
            .reset_index()
        )
        daily["model_variant"] = name
        error_rows.append(daily)
    errors_path = error_dir / "top_error_days.csv"
    pd.concat(error_rows, ignore_index=True).to_csv(errors_path, index=False)

    # Failure taxonomy for the most difficult individual half-hours. This is descriptive,
    # not causal: categories are context flags to support manual error analysis.
    taxonomy_rows: list[pd.DataFrame] = []
    train_demand = master.loc[master["split"] == "training", "national_demand_mw"].dropna()
    high_demand_cut = float(train_demand.quantile(0.95)) if len(train_demand) else float("inf")
    renewable_sum = (
        pd.to_numeric(master.get("embedded_wind_forecast_mw"), errors="coerce").fillna(0)
        + pd.to_numeric(master.get("embedded_solar_forecast_mw"), errors="coerce").fillna(0)
    )
    renewable_cut = float(renewable_sum.quantile(0.95)) if len(renewable_sum) else float("inf")
    context_cols = [
        "settlement_date", "settlement_period", "season", "is_weekend",
        "is_bank_holiday_ew", "is_bank_holiday_scotland",
        "embedded_wind_forecast_mw", "embedded_solar_forecast_mw",
        "is_normal_48_period_day",
    ]
    context_cols = [c for c in context_cols if c in master.columns]
    context = master[context_cols].copy()
    for name, frame in variants.items():
        f = frame.copy()
        f["abs_error_mw"] = (f["national_demand_mw"] - f["q50"]).abs()
        hard = f.nlargest(min(max(top_n * 48, 100), len(f)), "abs_error_mw")
        hard = hard.merge(context, on=["settlement_date", "settlement_period"], how="left")
        renewable = (
            pd.to_numeric(hard.get("embedded_wind_forecast_mw"), errors="coerce").fillna(0)
            + pd.to_numeric(hard.get("embedded_solar_forecast_mw"), errors="coerce").fillna(0)
        )
        hard["flag_high_demand"] = hard["national_demand_mw"] >= high_demand_cut
        hard["flag_high_renewable_forecast"] = renewable >= renewable_cut
        hard["flag_bank_holiday"] = (
            pd.to_numeric(hard.get("is_bank_holiday_ew"), errors="coerce").fillna(0).astype(int)
            | pd.to_numeric(hard.get("is_bank_holiday_scotland"), errors="coerce").fillna(0).astype(int)
        ).astype(bool)
        hard["flag_weekend"] = pd.to_numeric(hard.get("is_weekend"), errors="coerce").fillna(0).astype(bool)
        hard["flag_dst_irregular"] = (
            pd.to_numeric(hard.get("is_normal_48_period_day"), errors="coerce").fillna(1).astype(int) == 0
        )
        hard["model_variant"] = name
        taxonomy_rows.append(hard)
    taxonomy_path = error_dir / "error_taxonomy.csv"
    if taxonomy_rows:
        pd.concat(taxonomy_rows, ignore_index=True).to_csv(taxonomy_path, index=False)
    else:
        pd.DataFrame().to_csv(taxonomy_path, index=False)

    # Distribution-shift diagnostics.
    shift = distribution_shift_report(master, cfg["features"].get("shift_features", []))
    shift_path = metrics_dir / "distribution_shift.csv"
    shift.to_csv(shift_path, index=False)

    # Paired point-accuracy confidence intervals against seasonal naive.
    baseline = variants.get("seasonal_naive_split_conformal")
    boot_rows = []
    if baseline is not None:
        reps = int(cfg["evaluation"]["bootstrap_replicates"])
        seed = int(cfg["evaluation"]["bootstrap_seed"])
        for name, frame in variants.items():
            if name == "seasonal_naive_split_conformal":
                continue
            result = daily_block_bootstrap_difference(
                frame,
                baseline,
                metric="mae",
                replicates=reps,
                seed=seed,
            )
            boot_rows.append({"left_model": name, "right_model": "seasonal_naive", **result})
    bootstrap_path = bootstrap_dir / "mae_vs_seasonal_naive.csv"
    pd.DataFrame(boot_rows).to_csv(bootstrap_path, index=False)

    LOGGER.info("Evaluation complete: %d variants", len(variants))
    return {
        "overall_metrics": metrics_path,
        "slice_metrics": slices_path,
        "top_error_days": errors_path,
        "error_taxonomy": taxonomy_path,
        "distribution_shift": shift_path,
        "bootstrap": bootstrap_path,
    }
