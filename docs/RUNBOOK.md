# Operational Runbook

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Engineering smoke test

```bash
make test
make smoke
```

Synthetic smoke outputs are not research findings.

## 3. Security/static checks

```bash
make audit
```

## 4. Full NESO run

```bash
gbdf full-run --config configs/project.yaml
```

Do not manually modify generated metrics or predictions.

## 5. Inspect outputs

- `reports/data_quality.json`
- `reports/reproducibility_manifest.json`
- `reports/results_evidence_pack.md`
- `results/metrics/overall_metrics.csv`
- `results/metrics/slice_metrics.csv`
- `results/bootstrap/mae_vs_seasonal_naive.csv`
- `results/error_analysis/top_error_days.csv`
- `results/figures/`

## 6. Dashboard

```bash
streamlit run app/streamlit_app.py
```

## 7. Research log
Record: source retrieval date/checksums, any schema warning, tuning decision, frozen settings, test-opening date, bugs, reruns and review-requested changes.
