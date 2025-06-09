# Architecture

`acoustok` is a classic three-stage neural audio codec — encoder, residual vector
quantizer, decoder — wrapped in a tokenizer that speaks in discrete integers.

```
waveform  ──►  SEANet encoder  ──►  latent z  ──►  residual VQ  ──►  codes
(B,1,T)         conv downsample      (B,D,T/H)      n_q codebooks     (B,n_q,T/H)

codes  ──►  residual VQ decode  ──►  latent ẑ  ──►  SEANet decoder  ──►  waveform
```

## Encoder / decoder

The encoder is a [SEANet](https://arxiv.org/abs/2009.02095)-style fully
convolutional network (the same family used by SoundStream and EnCodec):

- an input convolution lifts the mono waveform to `n_filters` channels;
- a stack of **encoder blocks**, one per entry in `ratios`, each made of dilated
  residual units followed by a strided convolution that downsamples by that ratio
  and doubles the channel count;
- a final convolution projects to the latent `dimension`.

The decoder mirrors this exactly: it consumes the latent, upsamples through
transposed convolutions in reverse `ratios` order, and ends on a convolution back
to the waveform. The total downsampling factor is `hop_length = prod(ratios)`, so
the latent frame rate is `sample_rate / hop_length` (75 Hz with the defaults).

Length is preserved end to end. Every `SConv1d` pads its input so the output is
`ceil(length / stride)` frames, and every `SConvTranspose1d` trims the symmetric
amount back off — see [`modules/conv.py`](../acoustok/modules/conv.py). The
tokenizer pads inputs up to a multiple of `hop_length` and trims the
reconstruction back to the original length.

## Residual vector quantization

The latent is discretised by a stack of `num_quantizers` codebooks. Level *i*
quantizes the residual left by levels `0..i-1`:

```
residual ← z
for each codebook:
    q, code ← nearest_entry(residual)
    quantized ← quantized + q
    residual  ← residual - q
```

Summing more levels reconstructs the latent more faithfully, which is what makes
the bitrate adjustable: encoding with fewer levels simply stops the loop early.
Each codebook learns by **exponential moving average** of the encoder statistics
(rather than gradient descent on the embeddings), with Laplace-smoothed cluster
counts and dead-code expiry. Gradients reach the encoder through the
straight-through estimator plus a small commitment loss.

See [design-notes.md](design-notes.md) for *why* these choices, and
[api-reference.md](api-reference.md) for the call signatures.
