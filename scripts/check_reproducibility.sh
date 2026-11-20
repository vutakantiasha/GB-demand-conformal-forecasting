#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf data/raw_smoke data/interim_smoke data/processed_smoke artifacts_smoke results_smoke reports_smoke
gbdf full-run --config configs/smoke.yaml >/tmp/gbdf_smoke_a.log 2>&1
sha256sum results_smoke/metrics/overall_metrics.csv results_smoke/metrics/conformal_sensitivity.csv reports_smoke/results_evidence_pack.md > /tmp/gbdf_hash_a.txt
gbdf full-run --config configs/smoke.yaml >/tmp/gbdf_smoke_b.log 2>&1
sha256sum results_smoke/metrics/overall_metrics.csv results_smoke/metrics/conformal_sensitivity.csv reports_smoke/results_evidence_pack.md > /tmp/gbdf_hash_b.txt
diff -u /tmp/gbdf_hash_a.txt /tmp/gbdf_hash_b.txt
echo "Reproducibility check passed: key outputs are byte-identical across two full smoke runs."
