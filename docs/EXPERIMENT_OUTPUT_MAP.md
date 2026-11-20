# Experiment-to-Output Map

| Planned experiment | Implementation | Primary output |
|---|---|---|
| E1 Point benchmark | seasonal naive, LightGBM q50, TFT q50, NESO external | `results/metrics/overall_metrics.csv` |
| E2 Raw probabilistic forecast | LightGBM/TFT q05–q95 | `artifacts/calibration/*_raw.csv` |
| E3 Static calibration | static CQR | `artifacts/calibration/*_static_cqr.csv` |
| E4 Adaptive calibration | prequential adaptive CQR | `artifacts/calibration/*_adaptive_cqr.csv` |
| E5 Renewable feature ablation | full LightGBM vs `lightgbm_no_renewables` | overall + slice metrics |
| E6 Robustness slices | season, horizon band, weekend, demand decile, error taxonomy | `slice_metrics.csv`, `error_taxonomy.csv` |
| E7 Calibration sensitivity | 80/90/95% targets × gamma × rolling window | `conformal_sensitivity.csv` |

Additional evidence:
- paired day-block bootstrap: `results/bootstrap/mae_vs_seasonal_naive.csv`
- distribution shift: `results/metrics/distribution_shift.csv`
- LightGBM feature importance: `artifacts/models/lightgbm/feature_importance.csv`
- SHAP (full run): `results/metrics/lightgbm_shap_importance.csv` and figure
- top-error days: `results/error_analysis/top_error_days.csv`
