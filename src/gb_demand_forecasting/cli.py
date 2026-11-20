from __future__ import annotations

import argparse
import json
from pathlib import Path

from gb_demand_forecasting.orchestrator import acquire, full_run, prepare, train
from gb_demand_forecasting.utils.config import load_project_config
from gb_demand_forecasting.utils.logging import configure_logging
from gb_demand_forecasting.utils.reproducibility import seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gbdf",
        description="GB day-ahead demand forecasting and conformal calibration research pipeline.",
    )
    parser.add_argument("command", choices=["acquire", "prepare", "train", "full-run"])
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    configure_logging(args.verbose)
    root, cfg = load_project_config(args.config)
    seed_everything(int(cfg["project"]["random_seed"]))
    if args.command == "acquire":
        output = acquire(cfg, root)
    elif args.command == "prepare":
        output = prepare(cfg, root)
    elif args.command == "train":
        output = train(cfg, root)
    else:
        output = full_run(cfg, root)
    if isinstance(output, dict):
        print(json.dumps({k: str(v) for k, v in output.items()}, indent=2, default=str))
    elif isinstance(output, list):
        print("\n".join(str(x) for x in output))
    else:
        print(output)


if __name__ == "__main__":
    main()
