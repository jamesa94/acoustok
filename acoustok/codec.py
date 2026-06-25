"""The end-to-end neural codec: encoder, residual quantizer and decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CodecConfig
from .modules.seanet import SEANetDecoder, SEANetEncoder
from .quantization.residual_vq import ResidualVQ


class Codec(nn.Module):
    """Compress a waveform to discrete codes and back."""

    def __init__(self, config: CodecConfig | None = None) -> None:
        super().__init__()
        self.config = config or CodecConfig()
        c = self.config
        self.encoder = SEANetEncoder(
            channels=c.channels,
            dimension=c.dimension,
            n_filters=c.n_filters,
            ratios=c.ratios,
            kernel_size=c.kernel_size,
            last_kernel_size=c.last_kernel_size,
            dilations=c.dilations,
            activation=c.activation,
            norm=c.norm,
            pad_mode=c.pad_mode,
        )
        self.decoder = SEANetDecoder(
            channels=c.channels,
            dimension=c.dimension,
            n_filters=c.n_filters,
            ratios=c.ratios,
            kernel_size=c.kernel_size,
            last_kernel_size=c.last_kernel_size,
            dilations=c.dilations,
            activation=c.activation,
            norm=c.norm,
            pad_mode=c.pad_mode,
        )
        self.quantizer = ResidualVQ(
            num_quantizers=c.num_quantizers,
            dim=c.dimension,
            codebook_size=c.codebook_size,
            codebook_dim=c.codebook_dim,
            decay=c.decay,
            kmeans_init=c.kmeans_init,
            commitment_weight=c.commitment_weight,
            threshold_ema_dead=c.threshold_ema_dead,
        )

    @property
    def hop_length(self) -> int:
        return self.config.hop_length

    @property
    def frame_rate(self) -> float:
        return self.config.frame_rate

    @property
    def num_quantizers(self) -> int:
        return self.config.num_quantizers

    def _pad_to_hop(self, wav: torch.Tensor) -> torch.Tensor:
        length = wav.shape[-1]
        remainder = length % self.hop_length
        if remainder:
            wav = F.pad(wav, (0, self.hop_length - remainder))
        return wav

    def encode(self, wav: torch.Tensor, n_quantizers: int | None = None) -> torch.Tensor:
        """Waveform ``(B, C, T)`` -> codes ``(B, n_quantizers, frames)``."""
        wav = self._pad_to_hop(wav)
        emb = self.encoder(wav)
        return self.quantizer.encode(emb, n_quantizers)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes ``(B, n_quantizers, frames)`` -> waveform ``(B, C, frames*hop)``."""
        emb = self.quantizer.decode(codes)
        return self.decoder(emb)
