# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-26

### Added
- `AudioTokenizer.from_pretrained` and `Codec.from_pretrained` for loading saved
  checkpoints, with the architecture config travelling alongside the weights.
- Factorised codebooks (`CodecConfig.codebook_dim`) for quantizing in a
  lower-dimensional projected space.
- `flatten_codes` / `unflatten_codes` with per-level vocabulary offsetting for
  feeding codes to a language model.
- `acoustok` command-line interface (`info`, `encode`, `decode`).
- Benchmarks for the residual quantizer under `benchmarks/`.

### Changed
- Minimum supported Python is now 3.10.
- Dead-code expiry in the codebook now reseeds from the current batch.

## [0.2.0] - 2026-06-18

### Added
- `ResidualVQ` with adjustable active level count and bandwidth control.
- WAV and token (`.npz`) file I/O using only the standard library plus NumPy.
- Documentation set under `docs/` (architecture, usage, design notes, API).

### Fixed
- Encoder/decoder length mismatch on inputs that were not a multiple of the hop.

## [0.1.0] - 2026-06-10

### Added
- Initial SEANet-style encoder/decoder and EMA `VectorQuantizer`.
- `CodecConfig` and the end-to-end `Codec` module.
- Project scaffolding: packaging, linting, type-checking, CI, and tests.

[Unreleased]: https://github.com/jamesa94/acoustok/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jamesa94/acoustok/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jamesa94/acoustok/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jamesa94/acoustok/releases/tag/v0.1.0
