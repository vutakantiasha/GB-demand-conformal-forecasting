from pathlib import Path

from gb_demand_forecasting.data.synthetic import acquire_synthetic
from gb_demand_forecasting.data.prepare import prepare_dataset
from gb_demand_forecasting.utils.config import load_project_config


def test_synthetic_acquire_and_prepare(tmp_path):
    _, cfg = load_project_config(Path("configs/smoke.yaml"))
    cfg["paths"] = {k: str(tmp_path / v) for k, v in cfg["paths"].items()}
    root = Path("/")
    manifest = acquire_synthetic(cfg, root)
    assert manifest.exists()
    master = prepare_dataset(cfg, root)
    assert master.exists()
