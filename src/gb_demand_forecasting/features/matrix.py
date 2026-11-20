from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureImputer:
    medians: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str]) -> "FeatureImputer":
        medians: dict[str, float] = {}
        for col in features:
            values = pd.to_numeric(df[col], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            medians[col] = median
        return cls(medians=medians)

    def transform(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in features:
            values = pd.to_numeric(df[col], errors="coerce")
            out[col] = values.fillna(self.medians[col]).astype(float)
        return out


def primary_rows(df: pd.DataFrame, exclude_dst: bool = True) -> pd.DataFrame:
    out = df.copy()
    if exclude_dst and "is_normal_48_period_day" in out.columns:
        out = out[out["is_normal_48_period_day"] == 1]
    return out.reset_index(drop=True)


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].copy().reset_index(drop=True)


def add_demand_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out["demand_decile"] = pd.qcut(
            out["national_demand_mw"], 10, labels=False, duplicates="drop"
        )
    except ValueError:
        out["demand_decile"] = 0
    return out
