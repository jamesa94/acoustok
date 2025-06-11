# Design notes

A grab-bag of the decisions behind `acoustok` and why they went the way they did.

## Why EMA codebooks instead of plain straight-through

The simplest discrete bottleneck is a learnable embedding table trained purely
through the straight-through estimator (what several reference codecs do). It
works, but the codebook tends to collapse: a handful of entries win every
assignment and the rest go unused.

`acoustok` instead updates each codebook with an **exponential moving average** of
the encoder vectors assigned to it — the VQ-VAE-2 / SoundStream recipe. Cluster
counts are Laplace-smoothed so an unused entry never divides by zero, and entries
whose usage falls below `threshold_ema_dead` are **resampled** from the current
batch. The encoder is still trained by gradients (straight-through plus a small
commitment loss); only the embedding table itself is EMA-driven. In practice this
keeps codebook utilisation high without any auxiliary entropy loss.

## Why length-preserving convolutions

A codec that does not return samples aligned with its input is a constant source
of off-by-one pain downstream (windowing, loss masking, streaming). Rather than
sprinkle ad-hoc paddings through the model, all of the arithmetic lives in
`SConv1d` / `SConvTranspose1d`: the encoder produces `ceil(T / stride)` frames per
layer and the decoder inverts each stride exactly. The tokenizer pads to a
multiple of `hop_length` once and trims the reconstruction back. Everything in
between can ignore the problem.

## Why a factorised codebook is optional

Quantizing in a low-dimensional projected space (`codebook_dim < dimension`)
improves codebook usage and is cheap — it is the trick DAC uses. It is supported
(`CodecConfig.codebook_dim`) but **off by default** because the default model is
small enough that the extra projections are not worth the parameter count. Turn
it on for larger configs.

## Why standard-library audio I/O

The library targets an offline-by-default, reproducible workflow. Depending on
libsndfile (via `soundfile`) or the full `torchaudio` stack pulls in system
libraries that complicate exactly that. 16-bit PCM WAV through the `wave` module
covers the common case with zero system dependencies; the linear resampler is
deliberately basic and documented as such. Reach for `torchaudio`/`soxr` when you
need research-grade resampling or exotic formats.

## Random weights on construction

A freshly built `Codec` is architecture, not a trained model — its reconstructions
are noise. This is intentional and documented everywhere it matters: the library
ships the *recipe*, not pretrained checkpoints. The quantization and tokenization
paths are exact and deterministic regardless of training, which is what the test
suite pins down.
