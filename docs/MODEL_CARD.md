# Model Card

## Intended use
Academic evaluation of Great Britain national day-ahead electricity-demand forecasting and uncertainty calibration. Not approved for dispatch, trading, balancing or live grid decisions.

## Models

### Seasonal naive
Weekly same-horizon baseline.

### LightGBM quantile regression
Independent q05/q50/q95 gradient-boosted tree regressors trained on leakage-safe features. Hyperparameters selected on validation median MAE and frozen before calibration/test.

### Research Temporal Fusion Transformer
A self-contained PyTorch implementation containing TFT's principal mechanisms used in this study: variable selection, gated residual networks, recurrent encoder/decoder processing, gated residual paths, temporal multi-head attention and direct quantile outputs. The static-covariate path is omitted because the dataset contains one aggregate national series and no informative entity-level static covariates.

The model is trained as a fixed-origin 72-step decoder: 24 remaining half-hours of D-1 plus 48 half-hours of target day D. Only the final 48 horizons are evaluated.

## Limitations

- National aggregate demand only; no local/household inference.
- No weather forecast variables beyond embedded renewable forecasts in the default specification.
- Operational NESO forecasting has access to richer systems and expertise; direct superiority claims are inappropriate.
- Temporal shift can degrade conformal assumptions; adaptive calibration is evaluated, not assumed to solve all drift.
- DST days require separate treatment.
