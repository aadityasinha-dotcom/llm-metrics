.PHONY: install test lint fmt build clean

install:
	python -m pip install -e ".[dev,openai,langchain]"

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
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
