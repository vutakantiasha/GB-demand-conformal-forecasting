from datetime import date, timedelta

import pandas as pd

from gb_demand_forecasting.models.tft import DailyWindowDataset, Standardizer


def test_tft_fixed_origin_window_lengths():
    rows = []
    start = date(2025, 1, 1)
    for d in range(20):
        day = start + timedelta(days=d)
        for sp in range(1, 49):
            rows.append({
                "settlement_date": day,
                "settlement_period": sp,
                "national_demand_mw": 20000 + d * 10 + sp,
                "f1": float(sp),
                "f2": float(d),
            })
    frame = pd.DataFrame(rows)
    scaler = Standardizer.fit(frame.iloc[: 10 * 48], ["f1", "f2"])
    target = start + timedelta(days=12)
    ds = DailyWindowDataset(frame, [target], ["f1", "f2"], scaler, encoder_days=7, origin_period=24)
    assert len(ds) == 1
    item = ds[0]
    assert item["encoder"].shape == (336, 3)
    assert item["decoder"].shape == (72, 2)
    assert item["target"].shape == (72,)
