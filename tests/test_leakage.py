from datetime import date, timedelta

import numpy as np
import pandas as pd

from gb_demand_forecasting.data.prepare import _add_leakage_safe_history


def _frame(days=20):
    rows = []
    start = date(2025, 1, 1)
    for d in range(days):
        for sp in range(1, 49):
            rows.append({"settlement_date": start + timedelta(days=d), "settlement_period": sp, "national_demand_mw": 20000 + 100*d + sp})
    return pd.DataFrame(rows)


def test_lag1_is_masked_after_forecast_origin():
    out = _add_leakage_safe_history(_frame(), "12:15")
    late = out[out["settlement_period"] > 24]
    early = out[out["settlement_period"] <= 24]
    assert late["demand_lag_1d_mw"].isna().all()
    assert early.iloc[48:]["demand_lag_1d_mw"].notna().all()


def test_lag2_and_lag7_remain_available_after_warmup():
    out = _add_leakage_safe_history(_frame(), "12:15")
    mature = out[pd.to_datetime(out["settlement_date"]) >= pd.Timestamp("2025-01-10")]
    assert mature["demand_lag_2d_mw"].notna().all()
    assert mature["demand_lag_7d_mw"].notna().all()
