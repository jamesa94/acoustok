"""Residual vector quantization built from a stack of :class:`VectorQuantizer`.

Each level quantizes the residual left by the previous one, so summing the levels
reconstructs the input ever more closely.  The number of active levels can be
chosen at call time, which is exactly the knob bandwidth control turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn

from .vq import VectorQuantizer


@dataclass
class QuantizeResult:
    """Bundle returned by :meth:`ResidualVQ.forward`."""

    quantized: torch.Tensor
    codes: torch.Tensor  # (batch, n_quantizers, time)
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class ResidualVQ(nn.Module):
    """A cascade of vector quantizers operating on the encoder residual."""

    def __init__(
        self,
        num_quantizers: int,
        dim: int,
        codebook_size: int,
        **vq_kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.layers = nn.ModuleList(
            VectorQuantizer(dim, codebook_size, **vq_kwargs) for _ in range(num_quantizers)
        )

    def _resolve_n(self, n_quantizers: int | None) -> int:
        n = self.num_quantizers if n_quantizers is None else n_quantizers
        if not 1 <= n <= self.num_quantizers:
            raise ValueError(
                f"n_quantizers must be in [1, {self.num_quantizers}], got {n_quantizers}"
            )
        return n

    def forward(self, x: torch.Tensor, n_quantizers: int | None = None) -> QuantizeResult:
        n = self._resolve_n(n_quantizers)
        quantized_out = torch.zeros_like(x)
        residual = x
        all_codes: list[torch.Tensor] = []
        total_loss = x.new_zeros(())

        for layer in self.layers[:n]:
            quantized, codes, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_codes.append(codes)
            total_loss = total_loss + loss

        codes = torch.stack(all_codes, dim=1)
        metrics = {"num_quantizers": float(n)}
        return QuantizeResult(quantized_out, codes, total_loss / n, metrics)

    def encode(self, x: torch.Tensor, n_quantizers: int | None = None) -> torch.Tensor:
        """Return discrete codes ``(batch, n_quantizers, time)`` without losses."""
        n = self._resolve_n(n_quantizers)
        residual = x
        all_codes: list[torch.Tensor] = []
        for layer in self.layers[:n]:
            layer = cast(VectorQuantizer, layer)
            codes = layer.encode(residual)
            residual = residual - layer.decode(codes)
            all_codes.append(codes)
        return torch.stack(all_codes, dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct the quantized latent from codes ``(batch, n_q, time)``."""
        n = codes.shape[1]
        quantized: torch.Tensor | None = None
        for i in range(n):
            layer = cast(VectorQuantizer, self.layers[i])
            layer_out = layer.decode(codes[:, i])
            quantized = layer_out if quantized is None else quantized + layer_out
        assert quantized is not None  # n >= 1 by construction
        return quantized
