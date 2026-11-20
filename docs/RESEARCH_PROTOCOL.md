# Frozen Research Protocol

## Primary question
How reliably can machine-learning models produce 90% day-ahead prediction intervals for Great Britain national electricity demand under temporal distribution shift, and to what extent do static and adaptive conformal calibration improve coverage without excessive loss of interval sharpness?

## Forecast origin
12:15 Europe/London on the day before delivery. This is deliberately aligned with the operational day-ahead publication window rather than target-day midnight.

## Target
NESO National Demand (ND), MW, half-hourly settlement periods.

## Primary experiments

- E1: seasonal naive vs LightGBM median vs TFT median vs NESO operational point forecast.
- E2: LightGBM/TFT raw q05-q95 interval reliability.
- E3: raw interval vs static CQR.
- E4: static CQR vs adaptive CQR under chronological test drift.
- E5: feature ablation with and without embedded renewable forecasts.
- E6: robustness by season, horizon band, demand decile, weekday/weekend and highest-error days.
- E7: calibration-window/adaptation sensitivity. Any sensitivity setting is chosen without optimising on final test performance.

## Leakage policy

A value may be used as a feature only if it would have existed by the configured forecast origin. In particular, lag-1 same-settlement-period demand on D-1 is masked for settlement periods after the completed half-hours available by 12:15. Lag-2/lag-7/lag-14 features are safe. Embedded renewable forecasts are selected from the latest published snapshot at or before origin.

## Locked-test rule
The 2026 test set is opened only after model/configuration choices are frozen. If a genuine implementation bug is discovered after opening the test set, the bug and its impact must be logged and every affected experiment rerun through the same corrected pipeline.

## DST policy
46/50-period UK clock-change days are tagged. Primary 48-horizon results exclude those days. They may be reported separately as a sensitivity analysis but must not be silently coerced to 48 periods.

## Conformal interpretation
Coverage claims are empirical/marginal. No interval is described as giving a 90% conditional probability that one individual half-hour observation will lie inside it.
