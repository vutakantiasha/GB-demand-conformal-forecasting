import numpy as np
import pandas as pd

from gb_demand_forecasting.conformal.methods import adaptive_cqr, finite_sample_quantile, static_cqr


def _pred_frame(start, n_days):
    rng = np.random.default_rng(1)
    rows = []
    dates = pd.date_range(start, periods=n_days, freq="D")
    for d in dates:
        for sp in range(1, 49):
            y = 20000 + 1000*np.sin(2*np.pi*sp/48) + rng.normal(0, 250)
            rows.append({"settlement_date": d, "settlement_period": sp, "national_demand_mw": y, "q05": y-200, "q50": y+rng.normal(0,80), "q95": y+200})
    return pd.DataFrame(rows)


def test_finite_sample_quantile_is_finite():
    q = finite_sample_quantile(np.arange(100), 0.1)
    assert np.isfinite(q)
    assert q >= 89


def test_static_cqr_applies_symmetric_finite_correction():
    cal = _pred_frame("2025-01-01", 30)
    test = _pred_frame("2025-02-01", 5)
    out = static_cqr(cal, test)
    assert np.isfinite(out["conformal_correction_mw"]).all()
    raw_width = test["q95"].to_numpy() - test["q05"].to_numpy()
    expected = raw_width + 2 * out["conformal_correction_mw"].to_numpy()
    assert np.allclose(out["q95"].to_numpy() - out["q05"].to_numpy(), expected)


def test_adaptive_cqr_outputs_all_rows():
    cal = _pred_frame("2025-01-01", 30)
    test = _pred_frame("2025-02-01", 5)
    out = adaptive_cqr(cal, test, rolling_window=60)
    assert len(out) == len(test)
    assert out["adaptive_alpha_after_update"].between(0.01, 0.30).all()
