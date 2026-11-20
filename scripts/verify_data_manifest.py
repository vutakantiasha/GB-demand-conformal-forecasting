#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from gb_demand_forecasting.utils.io import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/raw/source_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("No data/raw/source_manifest.json; run acquisition first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed = 0
    for record in manifest.get("records", []):
        p = Path(record["local_path"])
        if not p.exists():
            p = root / "data/raw" / p.name
        actual = sha256_file(p)
        ok = actual == record["sha256"]
        print(f"{'OK' if ok else 'FAIL'} {p.name} {actual}")
        failed += int(not ok)
    if failed:
        raise SystemExit(f"{failed} checksum verification(s) failed")


if __name__ == "__main__":
    main()
