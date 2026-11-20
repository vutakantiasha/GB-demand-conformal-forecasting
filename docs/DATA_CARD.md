# Data Card

## Dataset family

National Energy System Operator (NESO) Open Data used to construct a Great Britain half-hourly day-ahead demand forecasting benchmark.

## Intended research use

- National Demand day-ahead point forecasting.
- Quantile forecasting and 90% prediction intervals.
- Static and adaptive conformal calibration.
- Comparison with a published operational NESO forecast.
- Temporal-distribution-shift and failure-mode analysis.

## Not intended for

- Live grid dispatch.
- Trading decisions.
- Household-level inference.
- Identification or profiling of individuals.
- Safety-critical operational control.

## Primary sources

1. NESO Historic Demand Data.
2. NESO Day Ahead Half Hourly Demand Forecast Performance.
3. NESO Embedded Solar and Wind Forecast archives.

## Target

`national_demand_mw`, constructed from NESO National Demand (`ND`) outturn.

## Unit of analysis

One Great Britain settlement half-hour, indexed by settlement date and settlement period.

## Forecast origin

12:15 Europe/London on the previous day. Features are filtered/masked according to whether they were observable by that origin.

## Known limitations

- NESO may revise historic demand and renewable outturn data retrospectively.
- UK clock changes create 46/50-period days; primary 48-horizon analysis excludes them.
- National aggregate demand is not household demand.
- Forecast publication timestamps and source schemas can evolve; live metadata preflight is required.
- Weather variables are not included in the frozen primary design; embedded renewable forecasts provide operationally available context but do not replace meteorological modelling.
- Conformal coverage is interpreted marginally/empirically, not as a conditional probability guarantee for each half-hour.

## Privacy and ethics

The sources are public, aggregate electricity-system data. No human participants, personal data, special-category data or identifiable individual-level records are processed by the pipeline.
