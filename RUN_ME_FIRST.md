# Run Me First

This repository is a reproducible research implementation for leakage-aware day-ahead Great Britain electricity-demand forecasting.

## 1. Requirements

- Python 3.11 or 3.12
- Git
- Internet access for NESO data acquisition
- Approximately 2.5 GB of free space for the raw embedded-renewables archives, plus working space

On macOS Apple Silicon, read `docs/MACOS_APPLE_SILICON.md` before the full run.

## 2. Create an isolated environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 3. Preflight checks

```bash
make doctor
make test
python scripts/security_audit.py
python scripts/validate_neso_metadata.py
```

Expected outcome: environment preflight passes, the test suite passes, the security audit reports no blocking issue, and the current NESO resource metadata resolves successfully.

## 4. Optional deterministic smoke run

```bash
gbdf full-run --config configs/smoke.yaml
```

This creates synthetic engineering outputs under the `_smoke` directories. These outputs are for implementation validation only.

## 5. Run the real NESO experiment

```bash
gbdf full-run --config configs/project.yaml
```

On macOS, `caffeinate` can be used to prevent sleep during a long acquisition/training run:

```bash
caffeinate -dimsu gbdf full-run --config configs/project.yaml
```

The real pipeline performs acquisition, checksumming, preprocessing, leakage-safe feature engineering, model fitting, conformal calibration, locked-test evaluation, robustness diagnostics and report generation.

## 6. Principal outputs

```text
artifacts/predictions/
artifacts/calibration/
results/metrics/
results/bootstrap/
results/error_analysis/
results/figures/
reports/data_quality.json
reports/training_summary.json
reports/reproducibility_manifest.json
reports/results_evidence_pack.md
```

## 7. Verify a locally downloaded raw-data manifest

After acquisition, verify the current raw files against their stored SHA-256 values:

```bash
python scripts/verify_data_manifest.py
```

## 8. Launch the results explorer

```bash
streamlit run app/streamlit_app.py
```

The app reads saved outputs only; it does not retrain or mutate the locked research protocol.

## 9. Interpretation rule

Never describe a nominal 90% conformal interval as a 90% conditional probability for one individual half-hour. The reported guarantee is empirical/marginal and must be interpreted with the documented temporal and subgroup diagnostics.
