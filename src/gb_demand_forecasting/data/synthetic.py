from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gb_demand_forecasting.utils.io import ensure_dir, sha256_file, write_json


def acquire_synthetic(cfg: dict, root: Path) -> Path:
    raw_dir = ensure_dir(root / cfg["paths"]["raw_dir"])
    syn = cfg["source"]["synthetic"]
    dates = pd.date_range(syn["start_date"], syn["end_date"], freq="D")
    seed = int(cfg["project"].get("random_seed", 2712))
    rng = np.random.default_rng(seed)
    demand_rows: list[dict] = []
    embedded_rows: list[dict] = []
    performance_rows: list[dict] = []
    for d_idx, date_ts in enumerate(dates):
        weekday = date_ts.dayofweek
        seasonal = 2400 * np.cos(2 * np.pi * (date_ts.dayofyear - 15) / 365.25)
        weekend = -1800 if weekday >= 5 else 0
        trend = 2.0 * d_idx
        for sp in range(1, 49):
            hour = (sp - 1) / 2
            morning = 4200 * np.exp(-((hour - 8.0) / 2.6) ** 2)
            evening = 6200 * np.exp(-((hour - 18.0) / 3.0) ** 2)
            overnight = -2600 * np.exp(-((hour - 3.0) / 2.8) ** 2)
            solar = max(0.0, 9000 * np.sin(np.pi * (hour - 6) / 14))
            wind = 4000 + 1600 * np.sin(2 * np.pi * (d_idx / 9 + sp / 48))
            base = 27500 + seasonal + weekend + trend + morning + evening + overnight
            noise = rng.normal(0, 480)
            demand = max(15000, base - 0.18 * solar - 0.08 * wind + noise)
            sd = date_ts.date()
            demand_rows.append(
                {
                    "SETTLEMENT_DATE": sd.isoformat(),
                    "SETTLEMENT_PERIOD": sp,
                    "ND": round(demand, 2),
                }
            )
            origin = datetime.combine(sd - timedelta(days=1), time(11, 0), tzinfo=timezone.utc)
            embedded_rows.append(
                {
                    "SETTLEMENT_DATE": sd.isoformat(),
                    "SETTLEMENT_PERIOD": sp,
                    "Forecast_Datetime": origin.isoformat(),
                    "EMBEDDED_WIND_FORECAST": round(wind + rng.normal(0, 250), 2),
                    "EMBEDDED_SOLAR_FORECAST": round(solar + rng.normal(0, 180), 2),
                }
            )
            neso = demand + rng.normal(60, 700)
            performance_rows.append(
                {
                    "Date": sd.isoformat(),
                    "Settlement Period": sp,
                    "Demand Forecast": round(neso, 2),
                    "Demand Outturn": round(demand + rng.normal(0, 15), 2),
                    "Publish Datetime": origin.isoformat(),
                }
            )
    files = {
        "historic_demand": raw_dir / "historic_demand_synthetic.csv",
        "embedded_forecast": raw_dir / "embedded_forecast_synthetic.csv",
        "day_ahead_performance": raw_dir / "day_ahead_performance_synthetic.csv",
    }
    pd.DataFrame(demand_rows).to_csv(files["historic_demand"], index=False)
    pd.DataFrame(embedded_rows).to_csv(files["embedded_forecast"], index=False)
    pd.DataFrame(performance_rows).to_csv(files["day_ahead_performance"], index=False)
    records = []
    for kind, path in files.items():
        records.append(
            {
                "kind": kind,
                "resource_id": "synthetic",
                "resource_name": path.name,
                "url": "synthetic://generated",
                "local_path": str(path.resolve().relative_to(root.resolve())),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    manifest = {"source": "synthetic", "records": records}
    return write_json(manifest, raw_dir / "source_manifest.json")
