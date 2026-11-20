from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gb_demand_forecasting.features.matrix import FeatureImputer
from gb_demand_forecasting.utils.io import ensure_dir


def lightgbm_shap_analysis(
    master: pd.DataFrame,
    model_file: Path,
    feature_schema_file: Path,
    out_csv: Path,
    out_png: Path,
    max_rows: int = 2000,
    seed: int = 2712,
) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP analysis requested but shap/matplotlib is unavailable") from exc

    schema = json.loads(feature_schema_file.read_text(encoding="utf-8"))
    features = list(schema["features"])
    imputer = FeatureImputer(medians={k: float(v) for k, v in schema["medians"].items()})
    test = master[master["split"] == "test"].copy()
    if len(test) > max_rows:
        test = test.sample(max_rows, random_state=seed).sort_index()
    x = imputer.transform(test, features)
    booster = lgb.Booster(model_file=str(model_file))
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    importance = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap_mw": np.mean(np.abs(values), axis=0),
        }
    ).sort_values("mean_abs_shap_mw", ascending=False)
    ensure_dir(out_csv.parent)
    ensure_dir(out_png.parent)
    importance.to_csv(out_csv, index=False)
    shap.summary_plot(values, x, feature_names=features, show=False, max_display=18)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_csv, out_png
