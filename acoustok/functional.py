"""Thin functional wrappers over a codec for one-off calls.

These exist for the common case where you already hold a :class:`Codec` and just
want a free function rather than a method — handy in notebooks and scripts.
"""

from __future__ import annotations

import torch

from .codec import Codec


def encode(
    wav: torch.Tensor,
    codec: Codec,
    bandwidth: float | None = None,
    n_quantizers: int | None = None,
) -> torch.Tensor:
    """Encode ``(B, C, T)`` audio to codes with an existing codec."""
    return codec.encode(wav, bandwidth=bandwidth, n_quantizers=n_quantizers)


def decode(codes: torch.Tensor, codec: Codec) -> torch.Tensor:
    """Decode codes ``(B, n_q, T)`` back to a waveform with an existing codec."""
    return codec.decode(codes)


def reconstruct(
    wav: torch.Tensor,
    codec: Codec,
    bandwidth: float | None = None,
    n_quantizers: int | None = None,
) -> torch.Tensor:
    """Encode then decode, returning the reconstructed waveform only."""
    recon, _, _ = codec(wav, bandwidth=bandwidth, n_quantizers=n_quantizers)
    return recon
