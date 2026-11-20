#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall -q src tests scripts app
pytest -q
python scripts/security_audit.py
gbdf full-run --config configs/smoke.yaml
