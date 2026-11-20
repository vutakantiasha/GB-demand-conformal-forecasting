from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from gb_demand_forecasting.data.schema import (
    normalize_demand,
    normalize_embedded,
    normalize_performance,
)
from gb_demand_forecasting.utils.io import ensure_dir, read_json, write_frame, write_json

LOGGER = logging.getLogger(__name__)


def _resolve_manifest_path(root: Path, cfg: dict, value: str) -> Path:
    p = Path(value)
    if p.exists():
        return p
    candidate = root / cfg["paths"]["raw_dir"] / p.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Manifest file not found: {value}")


def _load_embedded_latest_from_path(path: Path, cfg: dict, chunksize: int = 250_000) -> pd.DataFrame:
    """Stream a potentially large 0-14 day forecast archive and retain one leakage-safe
    snapshot per target date/settlement period. Peak memory is bounded by one CSV chunk
    plus the compact best-so-far table (~one row per target half-hour).
    """
    best = pd.DataFrame()
    tz_name = cfg["forecast"]["timezone"]
    origin_hhmm = cfg["forecast"]["origin_local_time"]
    for raw_chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = normalize_embedded(raw_chunk)
        if chunk.empty:
            continue
        unique_dates = chunk["settlement_date"].drop_duplicates()
        origin_map = {d: _origin_utc(d, tz_name, origin_hhmm) for d in unique_dates}
        chunk["forecast_origin_utc"] = chunk["settlement_date"].map(origin_map)
        chunk = chunk[chunk["forecast_datetime_utc"] <= chunk["forecast_origin_utc"]]
        if chunk.empty:
            continue
        chunk = (
            chunk.sort_values(["settlement_date", "settlement_period", "forecast_datetime_utc"])
            .drop_duplicates(["settlement_date", "settlement_period"], keep="last")
            .drop(columns=["forecast_origin_utc"])
        )
        if best.empty:
            best = chunk
        else:
            best = (
                pd.concat([best, chunk], ignore_index=True)
                .sort_values(["settlement_date", "settlement_period", "forecast_datetime_utc"])
                .drop_duplicates(["settlement_date", "settlement_period"], keep="last")
            )
    return best.reset_index(drop=True)


def _load_sources(cfg: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = root / cfg["paths"]["raw_dir"]
    manifest = read_json(raw_dir / "source_manifest.json")
    demand_frames: list[pd.DataFrame] = []
    embedded_frames: list[pd.DataFrame] = []
    perf_frames: list[pd.DataFrame] = []
    chunksize = int(cfg.get("source", {}).get("embedded_chunksize", 250_000))
    for record in manifest["records"]:
        path = _resolve_manifest_path(root, cfg, record["local_path"])
        kind = record["kind"]
        if kind == "historic_demand":
            demand_frames.append(normalize_demand(pd.read_csv(path, low_memory=False)))
        elif kind == "embedded_forecast":
            embedded_frames.append(_load_embedded_latest_from_path(path, cfg, chunksize=chunksize))
        elif kind == "day_ahead_performance":
            perf_frames.append(normalize_performance(pd.read_csv(path, low_memory=False)))
    if not demand_frames:
        raise RuntimeError("No historic demand files were loaded")
    demand = pd.concat(demand_frames, ignore_index=True).drop_duplicates(
        ["settlement_date", "settlement_period"], keep="last"
    )
    embedded = (
        pd.concat(embedded_frames, ignore_index=True)
        if embedded_frames
        else pd.DataFrame(
            columns=[
                "settlement_date",
                "settlement_period",
                "forecast_datetime_utc",
                "embedded_wind_forecast_mw",
                "embedded_solar_forecast_mw",
            ]
        )
    )
    if not embedded.empty:
        embedded = (
            embedded.sort_values(["settlement_date", "settlement_period", "forecast_datetime_utc"])
            .drop_duplicates(["settlement_date", "settlement_period"], keep="last")
            .reset_index(drop=True)
        )
    performance = (
        pd.concat(perf_frames, ignore_index=True)
        if perf_frames
        else pd.DataFrame(
            columns=[
                "settlement_date",
                "settlement_period",
                "neso_day_ahead_forecast_mw",
                "neso_performance_outturn_mw",
                "neso_publish_datetime_utc",
            ]
        )
    )
    return demand, embedded, performance


def _origin_utc(d: date, tz_name: str, hhmm: str) -> pd.Timestamp:
    hour, minute = [int(x) for x in hhmm.split(":")]
    local_tz = ZoneInfo(tz_name)
    local = datetime.combine(d - timedelta(days=1), time(hour, minute), tzinfo=local_tz)
    return pd.Timestamp(local).tz_convert("UTC")


def _bank_holiday_sets(years: list[int]) -> tuple[set[date], set[date]]:
    try:
        import holidays
    except ImportError:
        LOGGER.warning("holidays package unavailable; bank-holiday flags default to zero")
        return set(), set()

    # python-holidays exposes both country/subdivision and convenience-class APIs
    # across supported releases. Try the explicit subdivision API first, then classes.
    try:
        ew = holidays.country_holidays("GB", years=years, subdiv="England")
        sc = holidays.country_holidays("GB", years=years, subdiv="Scotland")
    except (AttributeError, TypeError, ValueError):
        try:
            ew = holidays.England(years=years)
            sc = holidays.Scotland(years=years)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("Unable to construct England/Scotland bank-holiday calendars") from exc
    return set(ew.keys()), set(sc.keys())


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["settlement_date"])
    out["day_of_week"] = dt.dt.dayofweek.astype(int)
    out["month"] = dt.dt.month.astype(int)
    out["day_of_year"] = dt.dt.dayofyear.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["sp_sin"] = np.sin(2 * np.pi * (out["settlement_period"] - 1) / 48.0)
    out["sp_cos"] = np.cos(2 * np.pi * (out["settlement_period"] - 1) / 48.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)
    years = sorted(dt.dt.year.unique().astype(int).tolist())
    ew, sc = _bank_holiday_sets(years)
    out["is_bank_holiday_ew"] = out["settlement_date"].isin(ew).astype(int)
    out["is_bank_holiday_scotland"] = out["settlement_date"].isin(sc).astype(int)
    month = out["month"]
    out["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["winter", "spring", "summer"],
        default="autumn",
    )
    return out


def _add_day_structure(df: pd.DataFrame, normal_periods: int) -> pd.DataFrame:
    out = df.copy()
    counts = out.groupby("settlement_date")["settlement_period"].nunique()
    out["periods_in_day"] = out["settlement_date"].map(counts).astype(int)
    out["is_normal_48_period_day"] = (out["periods_in_day"] == normal_periods).astype(int)
    return out


def _lag_map(df: pd.DataFrame, days: int) -> pd.DataFrame:
    tmp = df[["settlement_date", "settlement_period", "national_demand_mw"]].copy()
    tmp["settlement_date"] = tmp["settlement_date"].map(lambda d: d + timedelta(days=days))
    return tmp.rename(columns={"national_demand_mw": f"demand_lag_{days}d_mw"})


def _add_leakage_safe_history(df: pd.DataFrame, origin_hhmm: str) -> pd.DataFrame:
    out = df.copy()
    for days in [1, 2, 7, 14]:
        out = out.merge(_lag_map(df, days), on=["settlement_date", "settlement_period"], how="left")

    # At a 12:15 previous-day origin only fully completed settlement periods 1-24
    # from the previous day are treated as observable. Later same-period lag-1 values
    # are masked rather than leaked into the feature matrix.
    hour, minute = [int(x) for x in origin_hhmm.split(":")]
    completed_periods = int((hour * 60 + minute) // 30)
    out["lag_1d_available"] = (out["settlement_period"] <= completed_periods).astype(int)
    out.loc[out["lag_1d_available"] == 0, "demand_lag_1d_mw"] = np.nan

    # Rolling same-horizon history uses t-2 onward so every value exists before origin.
    ordered = out.sort_values(["settlement_period", "settlement_date"]).copy()
    grouped = ordered.groupby("settlement_period", sort=False)["national_demand_mw"]
    ordered["demand_rolling_mean_7d_mw"] = grouped.transform(
        lambda s: s.shift(2).rolling(7, min_periods=3).mean()
    )
    ordered["demand_rolling_std_7d_mw"] = grouped.transform(
        lambda s: s.shift(2).rolling(7, min_periods=3).std()
    )
    out = ordered.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    morning = df[df["settlement_period"].between(1, completed_periods)].groupby("settlement_date")[
        "national_demand_mw"
    ].agg(["mean", "max"])
    morning.index = [d + timedelta(days=1) for d in morning.index]
    out["prevday_morning_mean_mw"] = out["settlement_date"].map(morning["mean"])
    out["prevday_morning_max_mw"] = out["settlement_date"].map(morning["max"])
    return out


def _select_latest_embedded(
    base: pd.DataFrame, embedded: pd.DataFrame, tz_name: str, origin_hhmm: str
) -> pd.DataFrame:
    if embedded.empty:
        return base.assign(
            embedded_wind_forecast_mw=np.nan,
            embedded_solar_forecast_mw=np.nan,
        )
    origins = (
        base[["settlement_date"]]
        .drop_duplicates()
        .assign(
            forecast_origin_utc=lambda x: x["settlement_date"].map(
                lambda d: _origin_utc(d, tz_name, origin_hhmm)
            )
        )
    )
    candidates = embedded.merge(origins, on="settlement_date", how="inner")
    candidates = candidates[candidates["forecast_datetime_utc"] <= candidates["forecast_origin_utc"]]
    latest = (
        candidates.sort_values(
            ["settlement_date", "settlement_period", "forecast_datetime_utc"]
        )
        .drop_duplicates(["settlement_date", "settlement_period"], keep="last")
        [[
            "settlement_date",
            "settlement_period",
            "embedded_wind_forecast_mw",
            "embedded_solar_forecast_mw",
            "forecast_datetime_utc",
        ]]
        .rename(columns={"forecast_datetime_utc": "embedded_snapshot_utc"})
    )
    return base.merge(latest, on=["settlement_date", "settlement_period"], how="left")


def _assign_split(df: pd.DataFrame, split_cfg: dict) -> pd.DataFrame:
    out = df.copy()
    date_series = pd.to_datetime(out["settlement_date"])
    out["split"] = "outside"
    for split_name, (start, end) in split_cfg.items():
        mask = date_series.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        if (out.loc[mask, "split"] != "outside").any():
            raise ValueError(f"Overlapping split detected for {split_name}")
        out.loc[mask, "split"] = split_name
    return out


def _data_quality_report(df: pd.DataFrame, performance: pd.DataFrame) -> dict:
    key_dupes = int(df.duplicated(["settlement_date", "settlement_period"]).sum())
    invalid_demand = int(((df["national_demand_mw"] <= 0) | df["national_demand_mw"].isna()).sum())
    normal_days = int(
        df.loc[df["is_normal_48_period_day"] == 1, "settlement_date"].nunique()
    )
    irregular_days = int(
        df.loc[df["is_normal_48_period_day"] == 0, "settlement_date"].nunique()
    )
    report = {
        "rows": int(len(df)),
        "unique_days": int(df["settlement_date"].nunique()),
        "duplicate_keys": key_dupes,
        "invalid_target_rows": invalid_demand,
        "normal_48_period_days": normal_days,
        "irregular_46_or_50_period_days": irregular_days,
        "missing_fraction": {
            c: float(df[c].isna().mean())
            for c in [
                "demand_lag_2d_mw",
                "demand_lag_7d_mw",
                "embedded_wind_forecast_mw",
                "embedded_solar_forecast_mw",
                "neso_day_ahead_forecast_mw",
            ]
            if c in df.columns
        },
    }
    if "neso_performance_outturn_mw" in df.columns:
        common = df[["national_demand_mw", "neso_performance_outturn_mw"]].dropna()
        report["neso_outturn_vs_historic_demand_mae_mw"] = (
            float((common.iloc[:, 0] - common.iloc[:, 1]).abs().mean()) if len(common) else None
        )
    return report


def prepare_dataset(cfg: dict, root: Path) -> Path:
    demand, embedded, performance = _load_sources(cfg, root)
    demand = demand.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    demand = _add_day_structure(demand, int(cfg["forecast"]["normal_day_periods"]))
    demand = _add_calendar_features(demand)
    demand["forecast_origin_utc"] = demand["settlement_date"].map(
        lambda d: _origin_utc(
            d, cfg["forecast"]["timezone"], cfg["forecast"]["origin_local_time"]
        )
    )
    demand = _add_leakage_safe_history(demand, cfg["forecast"]["origin_local_time"])
    demand = _select_latest_embedded(
        demand, embedded, cfg["forecast"]["timezone"], cfg["forecast"]["origin_local_time"]
    )
    if not performance.empty:
        demand = demand.merge(
            performance,
            on=["settlement_date", "settlement_period"],
            how="left",
        )
    else:
        demand["neso_day_ahead_forecast_mw"] = np.nan
        demand["neso_performance_outturn_mw"] = np.nan
        demand["neso_publish_datetime_utc"] = pd.NaT
    demand = _assign_split(demand, cfg["splits"])

    report = _data_quality_report(demand, performance)
    if cfg["runtime"].get("fail_on_data_quality_error", True):
        if report["duplicate_keys"] > 0 or report["invalid_target_rows"] > 0:
            raise ValueError(f"Critical data quality failure: {report}")

    processed_dir = ensure_dir(root / cfg["paths"]["processed_dir"])
    path = write_frame(demand, processed_dir / "master_dataset.csv.gz")
    write_json(report, root / cfg["paths"]["reports_dir"] / "data_quality.json")
    LOGGER.info("Prepared %s rows -> %s", len(demand), path)
    return path
