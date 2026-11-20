from datetime import date

import pandas as pd

from gb_demand_forecasting.data.prepare import _select_latest_embedded


def test_latest_embedded_snapshot_respects_forecast_origin():
    base = pd.DataFrame({
        "settlement_date": [date(2025, 1, 2)],
        "settlement_period": [1],
    })
    embedded = pd.DataFrame({
        "settlement_date": [date(2025, 1, 2)] * 3,
        "settlement_period": [1, 1, 1],
        "forecast_datetime_utc": pd.to_datetime([
            "2025-01-01T10:00:00Z",
            "2025-01-01T12:00:00Z",
            "2025-01-01T13:00:00Z",
        ]),
        "embedded_wind_forecast_mw": [1000, 2000, 9999],
        "embedded_solar_forecast_mw": [100, 200, 999],
    })
    out = _select_latest_embedded(base, embedded, "Europe/London", "12:15")
    assert out.loc[0, "embedded_wind_forecast_mw"] == 2000
    assert out.loc[0, "embedded_solar_forecast_mw"] == 200
