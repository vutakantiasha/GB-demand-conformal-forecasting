#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
gbdf full-run --config configs/project.yaml
