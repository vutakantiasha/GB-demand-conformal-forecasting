# Automated Results Evidence Pack

**Project:** Reliable Day-Ahead Electricity Demand Forecasting for Great Britain Using Conformal Prediction under Temporal Distribution Shift

> This file is generated from saved experiment outputs. Synthetic smoke-test outputs are engineering validation only and must not be reported as research project findings.

## Data quality

- Rows: 66960
- Unique days: 1395
- Duplicate keys: 0
- Normal 48-period days: 1387
- Irregular clock-change days: 8

## Overall locked-test metrics

| Model variant | MAE MW | RMSE MW | Coverage | Width MW | Interval score |
|---|---:|---:|---:|---:|---:|
| neso_operational_external | 781.7 | 1062.7 | — | — | — |
| lightgbm_raw | 1226.8 | 1622.5 | 0.647 | 2920.4 | 9632.0 |
| lightgbm_static_cqr | 1226.8 | 1622.5 | 0.845 | 4564.0 | 7045.7 |
| lightgbm_adaptive_cqr | 1226.8 | 1622.5 | 0.905 | 5537.9 | 6974.4 |
| tft_raw | 1285.8 | 1689.1 | 0.799 | 3932.1 | 7375.3 |
| tft_static_cqr | 1285.8 | 1689.1 | 0.824 | 4210.9 | 7076.0 |
| tft_adaptive_cqr | 1285.8 | 1689.1 | 0.904 | 5279.7 | 6737.9 |
| lightgbm_no_renewables_raw | 1486.4 | 2014.8 | 0.704 | 4003.3 | 10234.1 |
| lightgbm_no_renewables_static_cqr | 1486.4 | 2014.8 | 0.860 | 5684.4 | 8153.7 |
| lightgbm_no_renewables_adaptive_cqr | 1486.4 | 2014.8 | 0.902 | 6697.0 | 8160.1 |
| seasonal_naive_split_conformal | 2148.2 | 2974.2 | 0.851 | 7862.8 | 13091.6 |

## Paired day-block bootstrap comparisons

| Left model | Comparator | MAE difference MW | 95% CI |
|---|---|---:|---:|
| lightgbm_raw | seasonal_naive | -921.4 | [-1088.9, -764.8] |
| lightgbm_static_cqr | seasonal_naive | -921.4 | [-1088.9, -764.8] |
| lightgbm_adaptive_cqr | seasonal_naive | -921.4 | [-1088.9, -764.8] |
| lightgbm_no_renewables_raw | seasonal_naive | -661.6 | [-810.5, -516.9] |
| lightgbm_no_renewables_static_cqr | seasonal_naive | -661.6 | [-810.5, -516.9] |
| lightgbm_no_renewables_adaptive_cqr | seasonal_naive | -661.6 | [-810.5, -516.9] |
| tft_raw | seasonal_naive | -811.5 | [-987.0, -657.5] |
| tft_static_cqr | seasonal_naive | -811.5 | [-987.0, -657.5] |
| tft_adaptive_cqr | seasonal_naive | -811.5 | [-987.0, -657.5] |
| neso_operational_external | seasonal_naive | -1366.6 | [-1534.8, -1213.9] |

## Distribution-shift diagnostics

| Feature | Standardized mean shift | KS statistic | Wasserstein |
|---|---:|---:|---:|
| national_demand_mw | -0.121 | 0.079 | 806.4 |
| demand_lag_7d_mw | -0.092 | 0.066 | 649.4 |
| embedded_wind_forecast_mw | 0.058 | 0.046 | 92.0 |
| embedded_solar_forecast_mw | 0.629 | 0.159 | 1373.1 |

## Interpretation guardrails

- Conformal coverage is reported empirically/marginally; a 90% interval is not described as a 90% conditional probability for an individual half hour.
- NESO is an external operational reference, not a target the research system is required to outperform.
- The locked test set must not be used to retune features, hyperparameters, calibration settings, or model families.
- Any post-test bug fix must be logged and all affected experiments rerun consistently.
