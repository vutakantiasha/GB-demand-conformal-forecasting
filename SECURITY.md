# Security Policy

## Scope

This is research software for offline/analytical forecasting. It is not designed to operate critical infrastructure, dispatch electricity, control market positions or expose a network service.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the repository owner rather than publishing exploit details in an issue before a fix is available.

## Repository controls

- No passwords, private keys, API tokens or credentials are required in source control.
- Raw NESO datasets are acquired over HTTPS from first-party resources and are checksum recorded.
- Non-HTTPS resource download locations are rejected.
- Model artefacts from untrusted third parties should not be deserialised.
- GitHub Actions run CI/static checks defined in `.github/workflows/`.
- `scripts/security_audit.py` provides a local security-oriented preflight.

## Dependency management

Dependencies are constrained in `pyproject.toml`, `requirements.txt` and `environment.yml`. Review automated dependency alerts before upgrading packages used in a locked research reproduction.
