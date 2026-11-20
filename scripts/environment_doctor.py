#!/usr/bin/env python3
"""Local environment and configuration preflight for the research pipeline."""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "torch": "torch",
    "yaml": "PyYAML",
    "requests": "requests",
    "matplotlib": "matplotlib",
}
OPTIONAL = {
    "holidays": "holidays",
    "shap": "shap",
    "streamlit": "streamlit",
    "plotly": "plotly",
}


def main() -> int:
    print(f"Python: {sys.version.split()[0]} | {platform.platform()}")
    if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
        print("ERROR: supported Python range is 3.11-3.13")
        return 2

    missing_required: list[str] = []
    for module, package in REQUIRED.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "installed")
            print(f"OK required: {package} ({version})")
        except ImportError:
            missing_required.append(package)
            print(f"MISSING required: {package}")

    for module, package in OPTIONAL.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "installed")
            print(f"OK optional: {package} ({version})")
        except ImportError:
            print(f"MISSING optional: {package}")

    try:
        import torch

        print(f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    for relative in [
        "configs/project.yaml",
        "configs/models/lightgbm.yaml",
        "configs/models/tft.yaml",
        "configs/models/conformal.yaml",
    ]:
        p = ROOT / relative
        print(f"{'OK' if p.exists() else 'MISSING'} config: {relative}")
        if not p.exists():
            missing_required.append(relative)

    if missing_required:
        print("\nEnvironment preflight FAILED.")
        return 1
    print("\nEnvironment preflight PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
