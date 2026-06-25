#!/usr/bin/env bash
# Run the full local check suite: lint, format, type-check, tests.
# Mirrors what CI does so you can catch problems before pushing.
set -euo pipefail

echo "==> ruff check"
ruff check acoustok tests

echo "==> ruff format --check"
ruff format --check acoustok tests

echo "==> mypy"
mypy acoustok

echo "==> pytest"
pytest --cov=acoustok --cov-report=term-missing

echo "All checks passed."
