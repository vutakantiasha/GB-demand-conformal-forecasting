import numpy as np
import pandas as pd

from gb_demand_forecasting.evaluation.metrics import compute_metrics


def test_metrics_known_values():
    df = pd.DataFrame({
        "national_demand_mw": [10.0, 20.0, 30.0],
        "q50": [11.0, 19.0, 31.0],
        "q05": [8.0, 18.0, 28.0],
        "q95": [12.0, 22.0, 32.0],
    })
    m = compute_metrics(df)
    assert np.isclose(m["mae_mw"], 1.0)
    assert np.isclose(m["picp"], 1.0)
    assert np.isclose(m["mpiw_mw"], 4.0)
