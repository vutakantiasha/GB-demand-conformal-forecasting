from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from gb_demand_forecasting.features.matrix import primary_rows
from gb_demand_forecasting.models.tft import train_tft
from gb_demand_forecasting.utils.config import load_yaml
from gb_demand_forecasting.utils.io import read_frame, write_json
from gb_demand_forecasting.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated TFT training worker")
    parser.add_argument("--master", required=True)
    parser.add_argument("--features-json", required=True)
    parser.add_argument("--tft-config", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--exclude-dst", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    configure_logging(False)
    frame = read_frame(Path(args.master))
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"]).dt.date
    frame = primary_rows(frame, exclude_dst=args.exclude_dst)
    features = json.loads(args.features_json)
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise ValueError("--features-json must decode to a list of strings")
    cfg = load_yaml(Path(args.tft_config))
    LOGGER.info("Starting isolated TFT worker | torch process is separate from LightGBM")
    pred, meta = train_tft(
        frame,
        features,
        cfg,
        Path(args.artifacts_dir),
        int(args.seed),
        device_name=args.device,
        num_workers=int(args.num_workers),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output, index=False)
    write_json(meta, Path(args.meta_output))
    LOGGER.info("Isolated TFT worker completed | predictions=%s", output)


if __name__ == "__main__":
    main()
