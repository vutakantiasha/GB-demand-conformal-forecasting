from pathlib import Path

import pandas as pd

from gb_demand_forecasting.data.prepare import _load_embedded_latest_from_path


def test_chunked_embedded_selection_keeps_latest_pre_origin(tmp_path: Path):
    path = tmp_path / "embedded.csv"
    pd.DataFrame({
        "SETTLEMENT_DATE": ["2025-01-02"] * 4,
        "SETTLEMENT_PERIOD": [1, 1, 2, 2],
        "Forecast_Datetime": [
            "2025-01-01T10:00:00Z",
            "2025-01-01T12:10:00Z",
            "2025-01-01T11:00:00Z",
            "2025-01-01T13:00:00Z",
        ],
        "EMBEDDED_WIND_FORECAST": [1000, 2000, 3000, 9999],
        "EMBEDDED_SOLAR_FORECAST": [100, 200, 300, 999],
    }).to_csv(path, index=False)
    cfg = {"forecast": {"timezone": "Europe/London", "origin_local_time": "12:15"}}
    out = _load_embedded_latest_from_path(path, cfg, chunksize=2).sort_values("settlement_period")
    assert out.iloc[0]["embedded_wind_forecast_mw"] == 2000
    assert out.iloc[1]["embedded_wind_forecast_mw"] == 3000
