UV ?= uv

.PHONY: check test test-upstream test-patched test-pinned tour relativity probe build
check:
	$(UV) run --no-sync ruff check .
	$(UV) run --no-sync ruff format --check .
	$(UV) run --no-sync mypy

test:
	$(UV) run --no-sync pytest

test-upstream:
	PYTHONPATH=$(HOME)/GitHub/xarray-upstream $(UV) run --no-sync pytest

test-patched:
	PYTHONPATH=$(HOME)/GitHub/xarray-patched-4 $(UV) run --no-sync pytest

# The pinned patch series from the fork, in its own environment (the stock .venv is untouched).
test-pinned:
	UV_PROJECT_ENVIRONMENT=.venv-patched $(UV) run --extra dev --group patched pytest

# Rebuild and execute the tour notebook (examples/xarrayrf_tour.ipynb).
tour:
	cd examples && UV_PROJECT_ENVIRONMENT=../.venv-patched $(UV) run --extra dev --group patched --group docs python ../tools/build_tour.py xarrayrf_tour.ipynb

# Rebuild and execute the relativity notebook (examples/relativity.ipynb).
relativity:
	cd examples && UV_PROJECT_ENVIRONMENT=../.venv-patched $(UV) run --extra dev --group patched --group docs python ../tools/build_relativity.py relativity.ipynb

probe:
	$(UV) run --no-sync python tools/binding_operation_probe.py

build:
	$(UV) build
	$(UV) run --no-sync python tools/check_distribution.py
