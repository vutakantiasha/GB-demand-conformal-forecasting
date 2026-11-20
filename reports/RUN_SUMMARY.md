# Real NESO Run Summary

## Completion status

**Completed successfully.** The full pipeline ran from NESO acquisition through locked-test evaluation and report generation.

## Run chronology

- Source mode: `neso`
- Historic demand resources: 2023, 2024, 2025, 2026
- Embedded wind/solar forecast archives: 2023, 2024, 2025, 2026
- Day-ahead operational benchmark: NESO half-hourly forecast performance resource
- Prepared dataset: **66,960 rows** across **1,395 unique days**
- Normal 48-period days: **1,387**
- Irregular clock-change days: **8**
- Duplicate settlement keys: **0**

## Model training

- LightGBM best validation MAE (full features): **1246.34 MW**
- LightGBM best validation MAE (no-renewables ablation): **1662.22 MW**
- TFT windows: **687 train / 111 validation / 236 calibration / 234 test**
- TFT best validation quantile loss: **0.04603**
- Evaluation variants: **11**

## Primary locked-test outcomes

- NESO operational reference MAE: **781.7 MW**
- LightGBM MAE: **1226.8 MW**
- TFT MAE: **1285.8 MW**
- Seasonal-naive MAE: **2148.2 MW**
- LightGBM adaptive CQR coverage: **90.55%**
- TFT adaptive CQR coverage: **90.36%**
- TFT adaptive CQR mean interval width: **5279.7 MW**

## Outputs generated

The completed run produced prediction tables, conformal calibration tables, overall/slice metrics, sensitivity analysis, distribution-shift diagnostics, day-block bootstrap results, error analysis, SHAP outputs, publication figures and a reproducibility manifest.

The raw terminal transcript is intentionally not versioned because it contains workstation-specific absolute paths and adds no scientific information beyond this structured run record.
