import pandas as pd

from gb_demand_forecasting.data.schema import normalize_demand, normalize_embedded, normalize_performance


def test_normalize_demand_aliases():
    raw = pd.DataFrame({"Settlement Date": ["01/01/2025"], "Settlement Period": [1], "National Demand": [22000]})
    out = normalize_demand(raw)
    assert out.loc[0, "settlement_period"] == 1
    assert out.loc[0, "national_demand_mw"] == 22000


def test_normalize_embedded_fields():
    raw = pd.DataFrame({
        "SETTLEMENT_DATE": ["2025-01-01"],
        "SETTLEMENT_PERIOD": [1],
        "Forecast_Datetime": ["2024-12-31T11:00:00Z"],
        "EMBEDDED_WIND_FORECAST": [3000],
        "EMBEDDED_SOLAR_FORECAST": [0],
    })
    out = normalize_embedded(raw)
    assert out.loc[0, "embedded_wind_forecast_mw"] == 3000
    assert str(out.loc[0, "forecast_datetime_utc"].tz) == "UTC"


def test_normalize_performance_fields():
    raw = pd.DataFrame({
        "Date": ["2025-01-01"],
        "Settlement Period": [1],
        "Demand Forecast": [24000],
        "Demand Outturn": [23800],
        "Publish Datetime": ["2024-12-31T12:00:00Z"],
    })
    out = normalize_performance(raw)
    assert out.loc[0, "neso_day_ahead_forecast_mw"] == 24000


def test_iso_dates_are_not_reinterpreted_day_first():
    raw = pd.DataFrame({
        "SETTLEMENT_DATE": ["2024-01-02", "2024-02-01", "2024-08-31"],
        "SETTLEMENT_PERIOD": [1, 1, 1],
        "ND": [20000, 21000, 22000],
    })
    out = normalize_demand(raw)
    assert [str(x) for x in out["settlement_date"]] == ["2024-01-02", "2024-02-01", "2024-08-31"]
