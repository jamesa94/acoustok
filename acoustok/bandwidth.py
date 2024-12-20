"""Helpers translating between target bitrate and quantizer count.

A residual quantizer with ``n`` active levels and a codebook of ``K`` entries
emits ``n`` integers per frame, each carrying ``log2(K)`` bits.  At a frame rate
``f`` that is a bitrate of ``n * f * log2(K)`` bits per second.  These functions
are the single place that arithmetic lives so the codec and CLI agree.
"""

from __future__ import annotations

import math


def bits_per_codebook(codebook_size: int) -> float:
    return math.log2(codebook_size)


def num_quantizers_for_bandwidth(
    frame_rate: float,
    codebook_size: int,
    bandwidth: float | None,
    max_quantizers: int,
) -> int:
    """Largest quantizer count whose bitrate does not exceed ``bandwidth``.

    ``bandwidth`` is given in kbps.  ``None`` (or a non-positive value) means
    "use every quantizer", which is the highest-quality setting.
    """
    if bandwidth is None or bandwidth <= 0:
        return max_quantizers
    bits_per_q = frame_rate * bits_per_codebook(codebook_size)
    n = int(math.floor(bandwidth * 1000.0 / bits_per_q))
    return max(1, min(n, max_quantizers))


def bandwidth_for_num_quantizers(
    frame_rate: float,
    codebook_size: int,
    n_quantizers: int,
) -> float:
    """Bitrate in kbps for a given number of active quantizers."""
    return n_quantizers * frame_rate * bits_per_codebook(codebook_size) / 1000.0


def token_rate(frame_rate: float, n_quantizers: int) -> float:
    """Discrete tokens emitted per second across all active quantizers."""
    return frame_rate * n_quantizers
