.PHONY: install test lint fmt build check-package ci clean help

help:
	@echo "install        dev install with extras"
	@echo "test           pytest"
	@echo "lint           ruff + mypy"
	@echo "fmt            ruff --fix + format"
	@echo "build          build wheel and sdist"
	@echo "check-package  assert py.typed ships and the extras stay optional"
	@echo "ci             everything CI runs, minus the 3.9-3.13 matrix"

install:
	python -m pip install -e ".[dev,openai,anthropic,langchain]"

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .
	python -m mypy

fmt:
	python -m ruff check --fix .
	python -m ruff format .

build:
	python -m pip install -q build
	rm -rf dist
	python -m build

check-package: build
	python -m pip install -q twine
	python -m twine check dist/*
	python scripts/check_wheel.py

# Everything the CI workflow runs on a single interpreter. The one thing it
# cannot reproduce is the 3.9-3.13 matrix, which needs interpreters this
# checkout does not have — push and let Actions cover that.
ci: lint test check-package
	@echo
	@echo "ci passed on $$(python -V 2>&1). Not covered here: the 3.9-3.13 matrix."

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
