# NESO Source Verification

The implementation intentionally uses first-party National Energy System Operator (NESO) resources rather than third-party mirrors.

## Historic Demand Data

Purpose: National Demand target/outturn history and settlement structure.

Portal: https://www.neso.energy/data-portal/historic-demand-data

Important handling rule: NESO notes that the dataset is populated 21 days in arrears and can receive retrospective corrections. Raw files are therefore checksum-versioned and never overwritten silently inside one research run.

## Day Ahead Half Hourly Demand Forecast Performance

Purpose: external operational half-hourly day-ahead forecast benchmark.

Portal: https://www.neso.energy/data-portal/day-ahead-half-hourly-demand-forecast-performance

The resource contains half-hourly national demand forecasts and outturns from April 2021 onwards, including publication datetime. It is used instead of the separate cardinal-point historic day-ahead archive because the research project evaluates 48 half-hour target horizons.

## Embedded Solar and Wind Forecasts

Purpose: known-future renewable covariates.

Portal: https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts

The archive contains half-hourly 0-14-day-ahead forecasts. The implementation does not simply take the last row in the file; it selects the latest forecast snapshot whose forecast timestamp is at or before the fixed previous-day research origin.

## Dataset discovery

The downloader calls NESO's documented CKAN-style action:

```text
https://api.neso.energy/api/3/action/datapackage_show?id=<dataset-id>
```

Resource IDs and filenames are discovered at runtime from metadata so the research code is less brittle than hard-coded download URLs.


## Resource download location

Current NESO `datapackage_show` payloads can expose the canonical HTTPS download location in the resource `path` field, while other CKAN-compatible payloads may populate `url`. The client therefore prefers a valid HTTPS `path` and falls back to a valid HTTPS `url`; non-HTTPS or missing locations are rejected.
