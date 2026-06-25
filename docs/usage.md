# Usage

## Install

```bash
pip install acoustok          # from PyPI
pip install -e ".[dev]"       # from a checkout, with the test/lint tools
```

The only runtime dependencies are `torch` and `numpy`. Audio I/O uses the
standard-library `wave` module, so there is no system dependency on libsndfile.

## Tokenizing audio

```python
import torch
from acoustok import AudioTokenizer

tok = AudioTokenizer()                  # 24 kHz, 8-level RVQ, random weights
wav = torch.randn(1, 1, 24000)          # (batch, channels, samples)

codes = tok.encode(wav, bandwidth=6.0)  # (batch, n_quantizers, frames)
recon = tok.decode(codes)               # (batch, channels, samples)
```

`encode` accepts a 1-D `(T,)` array, a 2-D `(channels, T)` array, or a 3-D
`(batch, channels, T)` tensor — NumPy or torch. Multi-channel input is downmixed
to mono. `decode` accepts codes with or without a batch dimension.

## Choosing a bitrate

Bandwidth is just a friendlier name for "how many quantizer levels to keep". Pick
it by `bandwidth` (kbps) or by an explicit `n_quantizers`:

```python
codes_low  = tok.encode(wav, bandwidth=1.5)   # 2 levels  -> 150 tokens/s
codes_high = tok.encode(wav, n_quantizers=8)  # 8 levels  -> 600 tokens/s
```

The first `k` levels of a high-bandwidth encoding are identical to a
`k`-level encoding, so you can train once and serve many bitrates.

## Files

```python
codes = tok.encode_file("speech.wav", bandwidth=3.0)  # resamples if needed
tok.tokens_to_file(codes, "speech.npz")               # compressed token archive
tok.decode_to_file(codes, "out.wav")
```

## Feeding a language model

```python
from acoustok import flatten_codes, unflatten_codes

stream = flatten_codes(codes, offset=True, codebook_size=1024)  # (batch, T*n_q)
# ... train / sample your LM over `stream` ...
codes_again = unflatten_codes(stream, num_quantizers=codes.shape[1],
                              offset=True, codebook_size=1024)
```

With `offset=True` each quantizer level lives in its own slice of the vocabulary,
so the flat stream indexes a single embedding table of size `n_q * codebook_size`.

## Loading trained weights

```python
from acoustok import Codec

codec = Codec.from_pretrained("acoustok-24khz.pt")
tok = AudioTokenizer(codec=codec)
```

Save with `codec.save(path)`; the architecture config travels with the weights.

## CLI

```bash
acoustok info                              # config + bandwidth grid
acoustok encode in.wav out.npz --bandwidth 6.0
acoustok decode out.npz out.wav
```
