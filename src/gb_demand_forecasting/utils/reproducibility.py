from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess  # nosec B404 -- fixed argv; read-only git provenance command
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed non-PyTorch libraries without importing heavy ML runtimes.

    PyTorch is deliberately seeded inside the isolated TFT worker. Keeping torch out of
    the orchestration process prevents macOS Apple-Silicon OpenMP runtime collisions
    when LightGBM and PyTorch are installed in the same environment.
    """
    random.seed(seed)  # noqa: S311  # non-cryptographic reproducibility seed
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stable_dict_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit(root: Path) -> str | None:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        return subprocess.check_output(  # noqa: S603  # fixed argv; no shell
            [git_executable, "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def package_versions(names: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def runtime_manifest(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "config_sha256": stable_dict_hash({k: v for k, v in cfg.items() if not k.startswith("_")}),
        "git_commit": git_commit(root),
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "torch",
                "PyYAML",
                "requests",
                "matplotlib",
                "holidays",
                "shap",
            ]
        ),
    }
