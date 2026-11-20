from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def distribution_shift_report(
    frame: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    train = frame[frame["split"] == "training"]
    test = frame[frame["split"] == "test"]
    for feature in features:
        if feature not in frame.columns:
            continue
        a = pd.to_numeric(train[feature], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(test[feature], errors="coerce").dropna().to_numpy(float)
        if len(a) < 20 or len(b) < 20:
            continue
        ks = ks_2samp(a, b)
        rows.append(
            {
                "feature": feature,
                "train_mean": float(np.mean(a)),
                "test_mean": float(np.mean(b)),
                "standardized_mean_shift": float((np.mean(b) - np.mean(a)) / (np.std(a) + 1e-9)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein_distance": float(wasserstein_distance(a, b)),
            }
        )
    return pd.DataFrame(rows)
