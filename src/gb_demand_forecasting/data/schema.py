from __future__ import annotations

import re

import numpy as np
import pandas as pd


def canonical_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip()).strip("_")
    return out.upper()


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [canonical_name(c) for c in out.columns]
    return out



def parse_date_series(series: pd.Series, utc: bool = False) -> pd.Series:
    """Parse mixed NESO date formats without misreading ISO YYYY-MM-DD as day-first."""
    raw = series.astype(str).str.strip()
    iso_mask = raw.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    if utc:
        result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
        if iso_mask.any():
            result.loc[iso_mask] = pd.to_datetime(raw.loc[iso_mask], errors="coerce", utc=True)
        if (~iso_mask).any():
            result.loc[~iso_mask] = pd.to_datetime(
                raw.loc[~iso_mask], errors="coerce", dayfirst=True, utc=True
            )
    else:
        result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        if iso_mask.any():
            result.loc[iso_mask] = pd.to_datetime(raw.loc[iso_mask], errors="coerce")
        if (~iso_mask).any():
            result.loc[~iso_mask] = pd.to_datetime(
                raw.loc[~iso_mask], errors="coerce", dayfirst=True
            )
    return result


def _first_existing(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for candidate in candidates:
        key = canonical_name(candidate)
        if key in df.columns:
            return key
    if required:
        raise KeyError(f"None of expected columns found: {candidates}. Got={list(df.columns)}")
    return None


def normalize_demand(df: pd.DataFrame) -> pd.DataFrame:
    d = canonicalize_columns(df)
    date_col = _first_existing(d, ["SETTLEMENT_DATE", "SETTLEMENT DATE"])
    sp_col = _first_existing(d, ["SETTLEMENT_PERIOD", "SETTLEMENT PERIOD"])
    nd_col = _first_existing(d, ["ND", "NATIONAL_DEMAND", "NATIONAL DEMAND"])
    indicator_col = _first_existing(
        d, ["FORECAST_ACTUAL_INDICATOR", "FORECAST OR ACTUAL INDICATOR"], required=False
    )
    out = pd.DataFrame(
        {
            "settlement_date": parse_date_series(d[date_col]).dt.date,
            "settlement_period": pd.to_numeric(d[sp_col], errors="coerce"),
            "national_demand_mw": pd.to_numeric(d[nd_col], errors="coerce"),
        }
    )
    if indicator_col:
        indicator = d[indicator_col].astype(str).str.strip().str.upper()
        out = out[indicator.isin(["A", "ACTUAL", "NAN", ""]) | indicator.isna()]
    out = out.dropna(subset=["settlement_date", "settlement_period", "national_demand_mw"])
    out["settlement_period"] = out["settlement_period"].astype(int)
    return out.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)


def normalize_embedded(df: pd.DataFrame) -> pd.DataFrame:
    d = canonicalize_columns(df)
    date_col = _first_existing(d, ["SETTLEMENT_DATE"])
    sp_col = _first_existing(d, ["SETTLEMENT_PERIOD"])
    forecast_dt_col = _first_existing(d, ["FORECAST_DATETIME", "FORECAST_DATETIME_UTC"])
    wind_col = _first_existing(d, ["EMBEDDED_WIND_FORECAST"])
    solar_col = _first_existing(d, ["EMBEDDED_SOLAR_FORECAST"])
    out = pd.DataFrame(
        {
            "settlement_date": parse_date_series(d[date_col], utc=True).dt.date,
            "settlement_period": pd.to_numeric(d[sp_col], errors="coerce"),
            "forecast_datetime_utc": pd.to_datetime(d[forecast_dt_col], errors="coerce", utc=True),
            "embedded_wind_forecast_mw": pd.to_numeric(d[wind_col], errors="coerce"),
            "embedded_solar_forecast_mw": pd.to_numeric(d[solar_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["settlement_date", "settlement_period", "forecast_datetime_utc"])
    out["settlement_period"] = out["settlement_period"].astype(int)
    return out.sort_values(
        ["settlement_date", "settlement_period", "forecast_datetime_utc"]
    ).reset_index(drop=True)


def normalize_performance(df: pd.DataFrame) -> pd.DataFrame:
    d = canonicalize_columns(df)
    date_col = _first_existing(d, ["DATE", "SETTLEMENT_DATE"], required=False)
    datetime_col = _first_existing(d, ["DATETIME"], required=False)
    sp_col = _first_existing(d, ["SETTLEMENT_PERIOD", "SETTLEMENT PERIOD"])
    forecast_col = _first_existing(d, ["DEMAND_FORECAST", "DEMAND FORECAST"])
    outturn_col = _first_existing(d, ["DEMAND_OUTTURN", "DEMAND OUTTURN"], required=False)
    publish_col = _first_existing(d, ["PUBLISH_DATETIME", "PUBLISH DATETIME"], required=False)
    if date_col:
        dates = parse_date_series(d[date_col]).dt.date
    elif datetime_col:
        dates = parse_date_series(d[datetime_col], utc=True).dt.date
    else:
        raise KeyError("Performance data require DATE or DATETIME")
    out = pd.DataFrame(
        {
            "settlement_date": dates,
            "settlement_period": pd.to_numeric(d[sp_col], errors="coerce"),
            "neso_day_ahead_forecast_mw": pd.to_numeric(d[forecast_col], errors="coerce"),
            "neso_performance_outturn_mw": (
                pd.to_numeric(d[outturn_col], errors="coerce") if outturn_col else np.nan
            ),
            "neso_publish_datetime_utc": (
                pd.to_datetime(d[publish_col], errors="coerce", utc=True)
                if publish_col
                else pd.NaT
            ),
        }
    )
    out = out.dropna(subset=["settlement_date", "settlement_period"])
    out["settlement_period"] = out["settlement_period"].astype(int)
    out = out.sort_values(["settlement_date", "settlement_period", "neso_publish_datetime_utc"])
    return out.drop_duplicates(["settlement_date", "settlement_period"], keep="last").reset_index(
        drop=True
    )
