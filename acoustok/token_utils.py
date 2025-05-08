"""Reshaping helpers that turn quantizer codes into LM-friendly streams.

Residual codes come out of the quantizer shaped ``(batch, n_quantizers, time)``.
Audio language models usually want a flat 1-D sequence per example.  The two
conventions below are the common ones; both are exact inverses of each other.
"""

from __future__ import annotations

import torch


def flatten_codes(
    codes: torch.Tensor,
    offset: bool = False,
    codebook_size: int | None = None,
) -> torch.Tensor:
    """Flatten ``(B, n_q, T)`` codes into ``(B, T * n_q)``, frame-major.

    The output interleaves quantizers within a frame: for each timestep ``t`` the
    sequence holds level ``0..n_q-1`` before moving to ``t + 1``.  This keeps all
    levels of one acoustic frame adjacent, which is what most acoustic LMs expect.

    When ``offset`` is set each quantizer level is shifted into its own slice of
    the vocabulary (level ``q`` occupies ``[q*K, (q+1)*K)``) so the whole stream
    indexes a single embedding table of size ``n_q * K``.
    """
    if codes.dim() != 3:
        raise ValueError(f"expected (B, n_q, T) codes, got shape {tuple(codes.shape)}")
    batch, n_q, time = codes.shape
    if offset:
        if codebook_size is None:
            raise ValueError("codebook_size is required when offset=True")
        shifts = torch.arange(n_q, device=codes.device).view(1, n_q, 1)
        codes = codes + shifts * codebook_size
    return codes.permute(0, 2, 1).reshape(batch, time * n_q)


def unflatten_codes(
    flat: torch.Tensor,
    num_quantizers: int,
    offset: bool = False,
    codebook_size: int | None = None,
) -> torch.Tensor:
    """Inverse of :func:`flatten_codes`."""
    if flat.dim() != 2:
        raise ValueError(f"expected (B, L) codes, got shape {tuple(flat.shape)}")
    batch, length = flat.shape
    if length % num_quantizers != 0:
        raise ValueError(f"length {length} is not divisible by num_quantizers {num_quantizers}")
    time = length // num_quantizers
    codes = flat.reshape(batch, time, num_quantizers).permute(0, 2, 1).contiguous()
    if offset:
        if codebook_size is None:
            raise ValueError("codebook_size is required when offset=True")
        shifts = torch.arange(num_quantizers, device=codes.device).view(1, num_quantizers, 1)
        codes = codes - shifts * codebook_size
    return codes


def num_tokens(codes: torch.Tensor) -> int:
    """Total number of discrete tokens in a code tensor (per example)."""
    if codes.dim() == 3:
        return int(codes.shape[1] * codes.shape[2])
    if codes.dim() == 2:
        return int(codes.shape[0] * codes.shape[1])
    raise ValueError(f"unexpected code rank {codes.dim()}")
