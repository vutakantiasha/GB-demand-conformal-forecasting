.PHONY: install install-dev doctor test audit smoke repro-check preflight run dashboard clean compile

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

doctor:
	python scripts/environment_doctor.py

compile:
	python -m compileall -q src tests scripts app

test: compile
	pytest -q

audit: compile
	python scripts/security_audit.py
	ruff check src tests scripts app
	bandit -r src -c pyproject.toml
	pip-audit -r requirements.txt

smoke:
	gbdf full-run --config configs/smoke.yaml

repro-check:
	./scripts/check_reproducibility.sh

preflight:
	python scripts/validate_neso_metadata.py

run:
	gbdf full-run --config configs/project.yaml

dashboard:
	streamlit run app/streamlit_app.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
