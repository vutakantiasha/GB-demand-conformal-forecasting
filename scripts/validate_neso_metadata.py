#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from gb_demand_forecasting.data.ckan import CKANClient
from gb_demand_forecasting.utils.config import load_project_config


def main() -> None:
    root, cfg = load_project_config(Path("configs/project.yaml"))
    del root
    source = cfg["source"]
    client = CKANClient(
        source["base_url"],
        timeout=int(source.get("timeout_seconds", 60)),
        retries=int(source.get("retries", 4)),
    )
    checks = []
    for year in source["years"]:
        checks.append(
            (
                source["datasets"]["historic_demand"],
                rf"Historic[_\s-]+Demand[_\s-]+Data[_\s-]+{year}$",
            )
        )
        checks.append(
            (
                source["datasets"]["embedded_forecasts"],
                rf"Embedded[_\s-]+Solar[_\s-]+and[_\s-]+Wind[_\s-]+Forecast[_\s-]+Archive[_\s-]+{year}$",
            )
        )
    for dataset, pattern in checks:
        resource = client.select_resource(dataset, pattern)
        print(f"OK {dataset}: {resource.get('name')} | {resource.get('id')} | {resource.get('format')}")
    package = client.package_show(source["datasets"]["day_ahead_performance"])
    csvs = [r for r in package.get("resources", []) if str(r.get("format", "")).upper() == "CSV"]
    if not csvs:
        raise SystemExit("No CSV resource in day-ahead performance dataset")
    print(f"OK day-ahead performance: {csvs[0].get('name')} | {csvs[0].get('id')}")


if __name__ == "__main__":
    main()
