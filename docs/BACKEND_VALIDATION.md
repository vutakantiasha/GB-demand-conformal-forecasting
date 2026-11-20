# Backend Validation Record

## Scope

This document records the engineering checks and the successful real-data execution that support the public research artefact.

## Real execution environment

The completed NESO experiment ran on Apple Silicon macOS with:

- Python 3.12.3
- LightGBM 4.7.0
- PyTorch 2.10.0
- NumPy 2.5.5
- pandas 2.3.3
- SciPy 1.18.1
- scikit-learn 1.9.1
- SHAP 0.54.1

The environment preflight passed before the real run.

## Engineering checks

1. Source/test/script/app bytecode compilation: **PASS**.
2. Local unit/integration suite before the real run: **16 passed**.
3. Public-release regression suite after portability/privacy patches: **19 passed**.
4. NESO live metadata validation: **PASS** for all required 2023–2026 resources and the day-ahead performance resource.
5. Deterministic synthetic end-to-end smoke pipeline: **PASS**.
6. LightGBM/TFT Apple-Silicon runtime isolation: **PASS**.
7. Full real NESO acquisition/preparation: **PASS**.
8. LightGBM full-feature and no-renewables training: **PASS**.
9. Isolated TFT training: **PASS**.
10. Eleven locked-test evaluation variants: **PASS**.
11. Core figures and SHAP summary generation: **PASS**.

## Important regression defects resolved

### Mixed date parsing

ISO-formatted dates are detected separately from day-first formats to avoid ambiguous parsing and compressed date ranges.

### NESO metadata endpoint

NESO uses `datapackage_show` for dataset metadata in this portal. The client and tests use that contract.

### NESO resource download field

Live NESO resource metadata exposes the canonical download location in `path` for relevant resources, while some CKAN-style payloads use `url`. The client now prefers a valid HTTPS `path` and falls back to a valid HTTPS `url` without weakening the transport check.

### Embedded forecast archive memory use

Large renewable forecast archives are streamed in bounded chunks and reduced to the latest snapshot available at or before the fixed forecast origin.

### CQR correction behaviour

CQR corrections can be negative when raw intervals are conservative. Tests therefore verify the conformal definition rather than assuming intervals can only widen.

### macOS native runtime collision

The TFT stage runs in an isolated Python subprocess so LightGBM and PyTorch do not need to coexist in the orchestration process. The actual model is retained; no substitute model is used.

## Real-run evidence

The final run prepared **66,960 rows**, trained the research models, completed **11 evaluation variants** and generated the retained metrics/figures. See `reports/RUN_SUMMARY.md` and `reports/results_evidence_pack.md`.

## Synthetic reproducibility evidence

Two complete synthetic smoke runs previously produced byte-identical key outputs with these SHA-256 values:

```text
e26e1f798edfe35829b235b356cec292d8cdbaa3cb9784c8b901ab293a7cb636  results_smoke/metrics/overall_metrics.csv
e84172f67ff17c6f95c5180a211b860e0a96be4fed25a011503214d5e68599d0  results_smoke/metrics/conformal_sensitivity.csv
85c25861ef14758d516f218fd571e2e9feb9354dd3bf5dd36ccda44c14fc6fb4  reports_smoke/results_evidence_pack.md
```

Synthetic outputs are engineering validation only and are not research findings.
