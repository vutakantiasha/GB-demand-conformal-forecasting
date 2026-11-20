from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str, sort_keys=True)
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_frame(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    suffixes = "".join(p.suffixes)
    if suffixes.endswith(".parquet"):
        try:
            df.to_parquet(p, index=False)
            return p
        except (ImportError, ModuleNotFoundError):
            p = p.with_suffix(".csv.gz")
    if "csv" in "".join(p.suffixes):
        df.to_csv(p, index=False)
    else:
        df.to_csv(p.with_suffix(".csv.gz"), index=False)
        p = p.with_suffix(".csv.gz")
    return p


def read_frame(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() and p.suffix == ".parquet":
        alt = p.with_suffix(".csv.gz")
        if alt.exists():
            p = alt
    if p.suffix == ".parquet":
        return pd.read_parquet(p, **kwargs)
    return pd.read_csv(p, **kwargs)
