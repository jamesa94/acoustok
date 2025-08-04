# acoustok

[![CI](https://github.com/jamesa94/acoustok/actions/workflows/ci.yml/badge.svg)](https://github.com/jamesa94/acoustok/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Neural audio codec and tokenizer for audio language models.**

`acoustok` turns raw waveforms into compact sequences of discrete **acoustic
tokens** with a convolutional encoder and residual vector quantization, and
reconstructs audio from those tokens with a matching decoder. It is designed for
the audio-LM workflow: adjustable-bitrate token streams, exact flatten/unflatten
helpers, a small typed API, and no heavyweight system dependencies.

```python
import torch
from acoustok import AudioTokenizer

tok = AudioTokenizer()                    # 24 kHz, 8-level residual quantizer
wav = torch.randn(1, 1, 24000)            # one second of audio (B, C, T)

codes = tok.encode(wav, bandwidth=6.0)    # (batch, n_quantizers, frames)
recon = tok.decode(codes)                 # back to a waveform
```

> [!NOTE]
> The bundled architecture ships with **random weights** — train it, or load a
> checkpoint, before expecting high-fidelity reconstruction. The quantization and
> tokenization plumbing is exact and deterministic regardless of training.

## Why acoustok

- **Codec *and* tokenizer.** A SEANet-style encoder/decoder plus residual VQ, with
  a high-level `AudioTokenizer` that handles shapes, devices and mono downmix.
- **Adjustable bitrate, one model.** Drop quantizer levels at inference time to
  trade fidelity for token rate — the first *k* levels of a high-bitrate encoding
  equal a *k*-level encoding.
- **LM-ready tokens.** `flatten_codes` interleaves residual levels per frame and
  can offset each level into its own vocabulary slice.
- **Stable codebooks.** EMA updates with Laplace smoothing and dead-code expiry,
  not bare straight-through.
- **Offline by default.** Only `torch` and `numpy`; audio I/O uses the standard
  library, so no libsndfile/`torchaudio` to install.
- **Properly engineered.** Typed throughout, `ruff` + `mypy` clean, tested on
  Python 3.10–3.12.

## Install

```bash
pip install acoustok
```

From a checkout, with the dev tools:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU build
pip install -e ".[dev]"
```

## How it works

```
waveform ─► SEANet encoder ─► latent ─► residual VQ ─► codes
(B,1,T)      conv downsample   (B,D,N)   n_q codebooks  (B,n_q,N)
                                                          │
recon ◄─ SEANet decoder ◄─ latent ◄─ residual VQ decode ◄─┘
```

The encoder downsamples by `hop_length = prod(ratios)` (320 by default → a 75 Hz
frame rate at 24 kHz). The residual quantizer stacks `num_quantizers` codebooks,
each refining the previous level's residual. Full details in
[docs/architecture.md](docs/architecture.md).

## Bitrate grid (defaults)

| Levels | Bitrate | Token rate |
| -----: | ------: | ---------: |
| 1 | 0.75 kbps | 75 tokens/s |
| 2 | 1.50 kbps | 150 tokens/s |
| 4 | 3.00 kbps | 300 tokens/s |
| 8 | 6.00 kbps | 600 tokens/s |

```bash
acoustok info   # prints the full grid for any config
```

## Documentation

- [Usage](docs/usage.md) — the everyday API, files, CLI, language-model streams
- [Architecture](docs/architecture.md) — encoder, decoder, residual quantizer
- [Design notes](docs/design-notes.md) — why the choices were made
- [API reference](docs/api-reference.md) — every public call signature
- [Examples](examples/) — four runnable scripts

## Development

```bash
pip install -e ".[dev]"
bash scripts/check.sh    # ruff + mypy + pytest, same as CI
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

The architecture follows the SoundStream / EnCodec / SEANet line of work on
neural audio codecs, and the residual-VQ tokenization is the representation used
by modern audio language models.

## License

[MIT](LICENSE) © Qu Wang
