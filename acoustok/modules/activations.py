"""Activation functions used by the encoder and decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class Snake(nn.Module):
    """Snake activation ``x + (1/a) * sin(a x) ** 2``.

    The periodic term gives the network a useful inductive bias for the
    quasi-periodic structure of audio (see the BigVGAN / DAC line of work).  The
    per-channel ``alpha`` is learnable.
    """

    def __init__(self, channels: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha + 1e-9
        return x + (1.0 / a) * torch.sin(a * x).pow(2)


def make_activation(name: str, channels: int | None = None) -> nn.Module:
    """Build an activation module by name.

    ``channels`` is only consulted by parametric activations (``snake``); the
    elementwise activations ignore it.
    """
    key = name.lower()
    if key == "snake":
        if channels is None:
            raise ValueError("snake activation requires the channel count")
        return Snake(channels)
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"unknown activation {name!r}")
