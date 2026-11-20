# macOS Apple Silicon runtime isolation

The orchestration process trains LightGBM in the main interpreter and launches the Temporal Fusion Transformer in a fresh Python subprocess.

This is intentional. On macOS Apple Silicon, LightGBM and PyTorch can load incompatible OpenMP runtimes into one process and cause hard segmentation faults when recurrent/attention kernels begin execution. A simple tensor test may pass while a full LSTM/TFT training loop crashes.

The isolated worker is `gb_demand_forecasting.models.tft_worker`. PyTorch seeding and deterministic settings are applied inside the worker, and the main orchestration process does not import PyTorch.

The project also pins macOS PyTorch to `<2.11` as an additional compatibility guard. Do not set `KMP_DUPLICATE_LIB_OK=TRUE` for research runs.
