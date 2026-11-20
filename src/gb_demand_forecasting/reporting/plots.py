from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gb_demand_forecasting.utils.io import ensure_dir


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(calibrated_files: list[Path], metrics_path: Path, out_dir: Path, rolling_days: int = 30) -> list[Path]:
    out_dir = ensure_dir(out_dir)
    metrics = pd.read_csv(metrics_path)
    frames: dict[str, pd.DataFrame] = {}
    for path in calibrated_files:
        df = pd.read_csv(path, parse_dates=["settlement_date"])
        name = str(df.get("model_variant", pd.Series([path.stem])).iloc[0])
        frames[name] = df
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    m = metrics.dropna(subset=["mae_mw"]).sort_values("mae_mw")
    ax.barh(m["model_variant"], m["mae_mw"])
    ax.set_xlabel("MAE (MW)")
    ax.set_title("Test-set point forecast accuracy")
    path = out_dir / "01_mae_comparison.png"
    _save(fig, path)
    written.append(path)

    interval_metrics = metrics.dropna(subset=["picp", "mpiw_mw"])
    if not interval_metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(interval_metrics["mpiw_mw"], interval_metrics["picp"])
        for row in interval_metrics.itertuples(index=False):
            ax.annotate(row.model_variant, (row.mpiw_mw, row.picp), fontsize=8)
        ax.axhline(0.90, linestyle="--")
        ax.set_xlabel("Mean prediction interval width (MW)")
        ax.set_ylabel("Empirical coverage")
        ax.set_title("Coverage-sharpness trade-off")
        path = out_dir / "02_coverage_vs_width.png"
        _save(fig, path)
        written.append(path)

    preferred = [n for n in frames if n.endswith("adaptive_cqr")]
    if not preferred:
        preferred = list(frames)
    if preferred:
        name = preferred[0]
        df = frames[name].copy().sort_values(["settlement_date", "settlement_period"])
        dates = sorted(df["settlement_date"].dt.date.unique())
        if dates:
            chosen = dates[len(dates) // 2]
            day = df[df["settlement_date"].dt.date == chosen]
            if len(day):
                fig, ax = plt.subplots(figsize=(11, 5))
                x = day["settlement_period"]
                ax.plot(x, day["national_demand_mw"], label="Actual")
                ax.plot(x, day["q50"], label="Median forecast")
                if day[["q05", "q95"]].notna().all().all():
                    ax.fill_between(x, day["q05"], day["q95"], alpha=0.2, label="90% interval")
                ax.set_xlabel("Settlement period")
                ax.set_ylabel("Demand (MW)")
                ax.set_title(f"{name}: example day {chosen}")
                ax.legend()
                path = out_dir / "03_example_day_interval.png"
                _save(fig, path)
                written.append(path)

        if df[["q05", "q95"]].notna().all(axis=1).any():
            df["inside"] = (
                (df["national_demand_mw"] >= df["q05"]) & (df["national_demand_mw"] <= df["q95"])
            ).astype(float)
            daily = df.groupby("settlement_date")["inside"].mean().sort_index()
            rolling = daily.rolling(rolling_days, min_periods=max(3, rolling_days // 4)).mean()
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(rolling.index, rolling.values)
            ax.axhline(0.90, linestyle="--")
            ax.set_ylim(0.5, 1.02)
            ax.set_ylabel("Rolling coverage")
            ax.set_title(f"{name}: rolling {rolling_days}-day empirical coverage")
            path = out_dir / "04_rolling_coverage.png"
            _save(fig, path)
            written.append(path)

    # Settlement-period MAE across point-capable variants.
    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = 0
    for name, df in frames.items():
        if df["q50"].notna().sum() == 0:
            continue
        temp = df.copy()
        temp["ae"] = (temp["national_demand_mw"] - temp["q50"]).abs()
        by_sp = temp.groupby("settlement_period")["ae"].mean()
        ax.plot(by_sp.index, by_sp.values, label=name)
        plotted += 1
    if plotted:
        ax.set_xlabel("Settlement period")
        ax.set_ylabel("MAE (MW)")
        ax.set_title("MAE by settlement period")
        ax.legend(fontsize=7, ncol=2)
        path = out_dir / "05_mae_by_settlement_period.png"
        _save(fig, path)
        written.append(path)
    else:
        plt.close(fig)
    return written
