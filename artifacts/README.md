# Runtime Artefacts

The full pipeline writes row-level model predictions, calibrated intervals and optional trained-model files under this directory. These files are intentionally **not versioned in the public repository** because they are reproducible, comparatively bulky generated outputs.

Runtime paths:

- `predictions/` — model prediction tables used by evaluation.
- `calibration/` — raw and conformalised interval tables.
- `models/` — optional model artefacts where enabled.

The public release retains compact locked metrics, figures, data-quality evidence, training summaries and provenance manifests under `results/` and `reports/`.
