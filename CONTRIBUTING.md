# Contributing to acoustok

Thanks for taking the time to contribute! This is a small project, so the process
is light.

## Development setup

```bash
git clone https://github.com/jamesa94/acoustok
cd acoustok
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU is fine
pip install -e ".[dev]"
pre-commit install
```

## The checks

CI runs three things on Python 3.10–3.12; please make sure they pass locally
before opening a pull request:

```bash
ruff check acoustok tests
ruff format --check acoustok tests
mypy acoustok
pytest
```

`pre-commit` runs ruff and mypy automatically on each commit.

## Guidelines

- **One concern per pull request.** Small, focused changes get reviewed faster.
- **Add a test** for any behaviour change. Tests live in `tests/` and mirror the
  module they cover (`test_<module>.py`).
- **Keep the public API small.** New top-level exports should earn their place.
- **Type everything.** Public functions carry annotations; `mypy` is part of CI.
- **Match the surrounding style.** Docstrings explain the *why*; comments are for
  non-obvious decisions, not narration.
- Update `CHANGELOG.md` under the `Unreleased` heading for user-facing changes.

## Reporting bugs

Open an issue with a minimal reproduction (audio shapes help!) and your
`acoustok` / `torch` / Python versions.
