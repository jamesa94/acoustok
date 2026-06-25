"""Reshaping helpers that turn quantizer codes into LM-friendly streams.

Residual codes come out of the quantizer shaped ``(batch, n_quantizers, time)``.
Audio language models usually want a flat 1-D sequence per example.
"""

from __future__ import annotations

import torch


def flatten_codes(codes: torch.Tensor) -> torch.Tensor:
    """Flatten ``(B, n_q, T)`` codes into ``(B, T * n_q)``, frame-major.

    The output interleaves quantizers within a frame: for each timestep ``t`` the
    sequence holds level ``0..n_q-1`` before moving to ``t + 1``.
    """
    if codes.dim() != 3:
        raise ValueError(f"expected (B, n_q, T) codes, got shape {tuple(codes.shape)}")
    batch, n_q, time = codes.shape
    return codes.permute(0, 2, 1).reshape(batch, time * n_q)


def unflatten_codes(flat: torch.Tensor, num_quantizers: int) -> torch.Tensor:
    """Inverse of :func:`flatten_codes`."""
    if flat.dim() != 2:
        raise ValueError(f"expected (B, L) codes, got shape {tuple(flat.shape)}")
    batch, length = flat.shape
    if length % num_quantizers != 0:
        raise ValueError(f"length {length} is not divisible by num_quantizers {num_quantizers}")
    time = length // num_quantizers
    return flat.reshape(batch, time, num_quantizers).permute(0, 2, 1).contiguous()
