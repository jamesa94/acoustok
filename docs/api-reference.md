# API reference

The public surface is everything exported from the top-level `acoustok` package.

## `AudioTokenizer`

High-level encode/decode of audio to acoustic tokens.

```python
AudioTokenizer(codec=None, config=None, device="cpu", bandwidth=None)
AudioTokenizer.from_pretrained(path, device="cpu", **kwargs)
```

| Method | Signature | Returns |
| --- | --- | --- |
| `encode` | `(wav, bandwidth=None, n_quantizers=None)` | codes `(B, n_q, T)` |
| `decode` | `(codes)` | waveform `(B, C, T)` |
| `encode_file` | `(path, bandwidth=None, n_quantizers=None)` | codes `(B, n_q, T)` |
| `decode_to_file` | `(codes, path)` | — |
| `tokens_to_file` | `(codes, path)` | — |
| `token_rate` | `(bandwidth=None, n_quantizers=None)` | tokens/second |
| `bandwidth_for` | `(n_quantizers)` | kbps |

Properties: `sample_rate`, `frame_rate`, `max_quantizers`.

## `Codec`

The underlying `nn.Module`.

```python
Codec(config: CodecConfig | None = None)
codec.encode(wav, bandwidth=None, n_quantizers=None) -> codes
codec.decode(codes) -> waveform
codec(wav, bandwidth=None, n_quantizers=None) -> (recon, codes, loss)
codec.save(path)
Codec.from_pretrained(path, map_location="cpu") -> Codec
```

Properties: `hop_length`, `frame_rate`, `num_quantizers`.

## `CodecConfig`

Dataclass describing the architecture. Key fields: `sample_rate`, `channels`,
`dimension`, `n_filters`, `ratios`, `codebook_size`, `codebook_dim`,
`num_quantizers`, `decay`, `commitment_weight`, `threshold_ema_dead`.
Derived properties: `hop_length`, `frame_rate`, `bits_per_codebook`,
`max_bandwidth`. Serialises with `to_dict()` / `from_dict()`.

## Quantization

```python
VectorQuantizer(dim, codebook_size, codebook_dim=None, decay=0.99, ...)
    .encode(x) -> codes ; .decode(codes) -> x ; .forward(x) -> (quantized, codes, loss)

ResidualVQ(num_quantizers, dim, codebook_size, **vq_kwargs)
    .encode(x, n_quantizers=None) -> (B, n_q, T)
    .decode(codes) -> quantized
    .forward(x, n_quantizers=None) -> QuantizeResult(quantized, codes, loss, metrics)
```

## Token utilities

```python
flatten_codes(codes, offset=False, codebook_size=None) -> (B, T * n_q)
unflatten_codes(flat, num_quantizers, offset=False, codebook_size=None) -> (B, n_q, T)
num_tokens(codes) -> int
```

## Bandwidth helpers

```python
num_quantizers_for_bandwidth(frame_rate, codebook_size, bandwidth, max_quantizers) -> int
bandwidth_for_num_quantizers(frame_rate, codebook_size, n_quantizers) -> float  # kbps
token_rate(frame_rate, n_quantizers) -> float                                   # tokens/s
```

## File I/O

```python
load_audio(path, target_sr=None, mono=False) -> (wav, sr)
save_audio(path, wav, sample_rate)
save_tokens(path, codes, **meta)
load_tokens(path) -> (codes, meta)
```
