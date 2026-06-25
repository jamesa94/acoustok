# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/jamesa94/acoustok/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jamesa94/acoustok/releases/tag/v0.1.0
