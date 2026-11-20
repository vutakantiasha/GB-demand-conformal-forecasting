from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from gb_demand_forecasting.features.matrix import FeatureImputer
from gb_demand_forecasting.utils.io import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


def _candidate_params(tune_cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    grid = tune_cfg.get("candidate_grid", {})
    if not grid:
        return []
    keys = sorted(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    rng = np.random.default_rng(seed)
    rng.shuffle(combos)
    return combos[: int(tune_cfg.get("trials", len(combos)))]


def tune_median_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    cfg: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], FeatureImputer, pd.DataFrame]:
    imputer = FeatureImputer.fit(train, features)
    x_train = imputer.transform(train, features)
    x_valid = imputer.transform(valid, features)
    y_train = train["national_demand_mw"].astype(float)
    y_valid = valid["national_demand_mw"].astype(float)
    base = dict(cfg["base_params"])
    base["random_state"] = seed
    tune_cfg = cfg.get("tuning", {})
    candidates = _candidate_params(tune_cfg, seed) if tune_cfg.get("enabled", True) else []
    if not candidates:
        candidates = [{}]
    rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params = base.copy()
    for idx, candidate in enumerate(candidates, start=1):
        params = {**base, **candidate, "objective": "quantile", "alpha": 0.5}
        model = lgb.LGBMRegressor(**params)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        score = float(mean_absolute_error(y_valid, pred))
        rows.append({"trial": idx, "validation_mae_mw": score, **candidate})
        if score < best_score:
            best_score = score
            best_params = {**base, **candidate}
    LOGGER.info("LightGBM tuning best validation MAE=%.2f MW", best_score)
    return best_params, imputer, pd.DataFrame(rows)


def fit_quantile_models(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    cfg: dict[str, Any],
    artifacts_dir: Path,
    seed: int,
    model_name: str = "lightgbm",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    best_params, _, tuning = tune_median_model(train, valid, features, cfg, seed)
    fit_data = pd.concat([train, valid], ignore_index=True)
    imputer = FeatureImputer.fit(fit_data, features)
    x_fit = imputer.transform(fit_data, features)
    y_fit = fit_data["national_demand_mw"].astype(float)
    predict_data = pd.concat([calibration, test], ignore_index=True)
    x_pred = imputer.transform(predict_data, features)
    pred = predict_data[
        ["settlement_date", "settlement_period", "split", "national_demand_mw"]
    ].copy()
    pred["model"] = model_name
    models_dir = ensure_dir(artifacts_dir / "models" / model_name)
    quantile_cols = {0.05: "q05", 0.5: "q50", 0.95: "q95"}
    feature_importance: pd.DataFrame | None = None
    for q in [float(x) for x in cfg.get("quantiles", [0.05, 0.5, 0.95])]:
        params = {
            **best_params,
            "objective": "quantile",
            "alpha": q,
            "random_state": seed,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(x_fit, y_fit)
        col = quantile_cols.get(round(q, 2), f"q{int(q * 100):02d}")
        pred[col] = model.predict(x_pred)
        model.booster_.save_model(str(models_dir / f"quantile_{q:.2f}.txt"))
        if abs(q - 0.5) < 1e-9:
            feature_importance = pd.DataFrame(
                {"feature": features, "gain_importance": model.booster_.feature_importance("gain")}
            ).sort_values("gain_importance", ascending=False)
    write_json({"features": features, "medians": imputer.medians}, models_dir / "feature_schema.json")
    write_json(best_params, models_dir / "best_params.json")
    tuning.to_csv(models_dir / "tuning_trials.csv", index=False)
    if feature_importance is not None:
        feature_importance.to_csv(models_dir / "feature_importance.csv", index=False)
    metadata = {"best_params": best_params, "features": features}
    return pred, metadata
