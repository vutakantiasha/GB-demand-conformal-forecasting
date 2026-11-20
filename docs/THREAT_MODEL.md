# Technical Threat Model

| Threat | Control |
|---|---|
| Dataset silently revised upstream | SHA-256 source manifest and retrieval timestamp |
| Target leakage | Fixed forecast origin, lag gating, forecast-snapshot cutoff, tests |
| Test-set overfitting | Chronological locked split and frozen protocol |
| Unsafe deserialisation | LightGBM saved in native text; never load untrusted PyTorch weights |
| Supply-chain vulnerability | pinned version ranges, `pip-audit`, Dependabot/CI-friendly requirements |
| Hard-coded secrets | no credentials required; SAST secret-pattern scan; pre-commit detect-secrets |
| Shell injection | no `shell=True`; downloader only accepts HTTPS resource URLs |
| YAML code execution | `yaml.safe_load` only |
| Research-output tampering | generated evidence pack and reproducibility manifest |
