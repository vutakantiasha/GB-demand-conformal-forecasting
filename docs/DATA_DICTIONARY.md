# Data Dictionary

## Core keys

| Field | Meaning |
|---|---|
| `settlement_date` | UK electricity-market settlement date |
| `settlement_period` | Half-hour settlement period; normally 1–48 |
| `national_demand_mw` | NESO National Demand target in MW |
| `forecast_origin_utc` | Fixed D-1 12:15 Europe/London origin converted to UTC |
| `split` | training / validation / calibration / test |

## Calendar features

`sp_sin`, `sp_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `is_weekend`, `is_bank_holiday_ew`, `is_bank_holiday_scotland`, `season`.

## Historical demand features

| Field | Availability rule |
|---|---|
| `demand_lag_1d_mw` | Present only when same-period D-1 outturn completed by origin |
| `lag_1d_available` | Explicit missingness/availability indicator |
| `demand_lag_2d_mw` | Safe same-period D-2 outturn |
| `demand_lag_7d_mw` | Safe weekly seasonal lag |
| `demand_lag_14d_mw` | Safe fortnight seasonal lag |
| `demand_rolling_mean_7d_mw` | Same-period rolling history starting at D-2 |
| `demand_rolling_std_7d_mw` | Same-period rolling variability starting at D-2 |
| `prevday_morning_mean_mw` | D-1 completed SP1–24 mean |
| `prevday_morning_max_mw` | D-1 completed SP1–24 maximum |

## Known-future operational covariates

`embedded_wind_forecast_mw` and `embedded_solar_forecast_mw` are selected from the most recent archive snapshot available no later than forecast origin. `embedded_snapshot_utc` records the exact chosen source timestamp.

## External benchmark

`neso_day_ahead_forecast_mw` comes from NESO's Day Ahead Half Hourly Demand Forecast Performance dataset. `neso_performance_outturn_mw` is retained only as a consistency diagnostic; the project target remains Historic Demand Data ND.
