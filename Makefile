.PHONY: help install lint format type test check clean

help:
	@echo "install  install the package with dev dependencies"
	@echo "lint     run ruff (lint + format check)"
	@echo "format   auto-format with ruff"
	@echo "type     run mypy"
	@echo "test     run pytest with coverage"
	@echo "check    lint + type + test (what CI runs)"
	@echo "clean    remove caches and build artifacts"

install:
	pip install -e ".[dev]"

lint:
	ruff check acoustok tests
	ruff format --check acoustok tests

format:
	ruff check --fix acoustok tests
	ruff format acoustok tests

type:
	mypy acoustok

test:
	pytest --cov=acoustok --cov-report=term-missing

check: lint type test

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
