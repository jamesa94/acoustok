"""Configuration object describing a codec architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _prod(values: tuple[int, ...]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


@dataclass
class CodecConfig:
    """Hyper-parameters for :class:`acoustok.codec.Codec`."""

    sample_rate: int = 24000
    channels: int = 1

    dimension: int = 128
    n_filters: int = 32
    ratios: tuple[int, ...] = (8, 5, 4, 2)
    kernel_size: int = 7
    last_kernel_size: int = 7
    dilations: tuple[int, ...] = (1, 3, 9)
    activation: str = "elu"
    norm: str = "weight_norm"
    pad_mode: str = "reflect"

    codebook_size: int = 1024
    codebook_dim: int | None = None
    num_quantizers: int = 8
    decay: float = 0.99
    kmeans_init: bool = False
    commitment_weight: float = 1.0
    threshold_ema_dead: float = 2.0

    target_bandwidths: tuple[float, ...] = field(default_factory=lambda: (1.5, 3.0, 6.0))

    @property
    def hop_length(self) -> int:
        """Total temporal downsampling factor of the encoder."""
        return _prod(self.ratios)

    @property
    def frame_rate(self) -> float:
        """Latent frames produced per second of audio."""
        return self.sample_rate / self.hop_length

    @property
    def bits_per_codebook(self) -> float:
        return math.log2(self.codebook_size)

    @property
    def max_bandwidth(self) -> float:
        """Bitrate in kbps when all quantizers are active."""
        bits = self.frame_rate * self.num_quantizers * self.bits_per_codebook
        return bits / 1000.0
