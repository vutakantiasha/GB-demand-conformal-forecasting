from __future__ import annotations

from pathlib import Path

import pandas as pd

from gb_demand_forecasting.utils.io import ensure_dir


def _fmt(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_results_report(
    cfg: dict,
    metrics_path: Path,
    bootstrap_path: Path,
    shift_path: Path,
    data_quality_path: Path,
    out_path: Path,
) -> Path:
    ensure_dir(out_path.parent)
    metrics = pd.read_csv(metrics_path)
    bootstrap = pd.read_csv(bootstrap_path) if bootstrap_path.exists() else pd.DataFrame()
    shift = pd.read_csv(shift_path) if shift_path.exists() else pd.DataFrame()
    import json

    quality = json.loads(data_quality_path.read_text(encoding="utf-8")) if data_quality_path.exists() else {}
    lines = [
        "# Automated Results Evidence Pack",
        "",
        f"**Project:** {cfg['project']['title']}",
        "",
        "> This file is generated from saved experiment outputs. Synthetic smoke-test outputs are engineering validation only and must not be reported as research project findings.",
        "",
        "## Data quality",
        "",
        f"- Rows: {quality.get('rows', '—')}",
        f"- Unique days: {quality.get('unique_days', '—')}",
        f"- Duplicate keys: {quality.get('duplicate_keys', '—')}",
        f"- Normal 48-period days: {quality.get('normal_48_period_days', '—')}",
        f"- Irregular clock-change days: {quality.get('irregular_46_or_50_period_days', '—')}",
        "",
        "## Overall locked-test metrics",
        "",
        "| Model variant | MAE MW | RMSE MW | Coverage | Width MW | Interval score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model_variant} | {_fmt(row.mae_mw,1)} | {_fmt(row.rmse_mw,1)} | "
            f"{_fmt(getattr(row,'picp',float('nan')),3)} | {_fmt(getattr(row,'mpiw_mw',float('nan')),1)} | "
            f"{_fmt(getattr(row,'interval_score',float('nan')),1)} |"
        )
    lines.extend(["", "## Paired day-block bootstrap comparisons", ""])
    if bootstrap.empty:
        lines.append("No bootstrap comparisons were available.")
    else:
        lines.append("| Left model | Comparator | MAE difference MW | 95% CI |")
        lines.append("|---|---|---:|---:|")
        for row in bootstrap.itertuples(index=False):
            lines.append(
                f"| {row.left_model} | {row.right_model} | {_fmt(row.estimate,1)} | "
                f"[{_fmt(row.ci_lower,1)}, {_fmt(row.ci_upper,1)}] |"
            )
    lines.extend(["", "## Distribution-shift diagnostics", ""])
    if shift.empty:
        lines.append("No shift diagnostics were available.")
    else:
        lines.append("| Feature | Standardized mean shift | KS statistic | Wasserstein |")
        lines.append("|---|---:|---:|---:|")
        for row in shift.itertuples(index=False):
            lines.append(
                f"| {row.feature} | {_fmt(row.standardized_mean_shift)} | {_fmt(row.ks_statistic)} | {_fmt(row.wasserstein_distance,1)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Conformal coverage is reported empirically/marginally; a 90% interval is not described as a 90% conditional probability for an individual half hour.",
            "- NESO is an external operational reference, not a target the research system is required to outperform.",
            "- The locked test set must not be used to retune features, hyperparameters, calibration settings, or model families.",
            "- Any post-test bug fix must be logged and all affected experiments rerun consistently.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
