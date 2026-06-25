"""Configuration object describing a codec architecture.

The config is deliberately plain data (a dataclass) so it can be serialised
alongside a checkpoint and round-tripped without importing torch.  Everything a
model needs to be rebuilt lives here; the model code reads from it and never
hard-codes architecture constants.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _prod(values: tuple[int, ...]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


@dataclass
class CodecConfig:
    """Hyper-parameters for :class:`acoustok.codec.Codec`.

    The defaults describe a 24 kHz, 75 Hz-frame-rate speech codec with an
    eight-level residual quantizer and a 1024-entry codebook — i.e. a maximum
    bitrate of 6 kbps.  Pass a smaller config (fewer filters, shorter ratios)
    for unit tests or quick experiments.
    """

    sample_rate: int = 24000
    channels: int = 1

    # Encoder / decoder geometry.
    dimension: int = 128
    n_filters: int = 32
    ratios: tuple[int, ...] = (8, 5, 4, 2)
    kernel_size: int = 7
    last_kernel_size: int = 7
    dilations: tuple[int, ...] = (1, 3, 9)
    activation: str = "elu"
    norm: str = "weight_norm"
    pad_mode: str = "reflect"

    # Residual vector quantizer.
    codebook_size: int = 1024
    codebook_dim: int | None = None
    num_quantizers: int = 8
    decay: float = 0.99
    kmeans_init: bool = False
    commitment_weight: float = 1.0
    threshold_ema_dead: float = 2.0

    # Bandwidths (kbps) advertised by the model; informational only.
    target_bandwidths: tuple[float, ...] = field(default_factory=lambda: (1.5, 3.0, 6.0))

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if len(self.ratios) == 0:
            raise ValueError("ratios must contain at least one stride")
        if any(r <= 0 for r in self.ratios):
            raise ValueError("every ratio must be a positive integer")
        if self.codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1")
        if self.num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive")
        # Normalise sequence fields to tuples so configs compare equal regardless
        # of whether they were built from lists (e.g. loaded from JSON).
        self.ratios = tuple(int(r) for r in self.ratios)
        self.dilations = tuple(int(d) for d in self.dilations)
        self.target_bandwidths = tuple(float(b) for b in self.target_bandwidths)

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodecConfig:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
