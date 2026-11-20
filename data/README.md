# Data

Raw NESO datasets are intentionally **not redistributed** in this repository.

`raw/source_manifest.json` preserves the exact resource identifiers, first-party download URLs, retrieval timestamps, file sizes and SHA-256 hashes from the locked research run. Running the acquisition stage recreates the local raw files under `data/raw/`.

Generated interim/processed datasets are ignored by Git because they can be reproduced from the acquisition and preparation pipeline.
