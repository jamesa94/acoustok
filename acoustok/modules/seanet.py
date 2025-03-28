"""SEANet-style convolutional encoder and decoder.

The encoder downsamples a waveform through a cascade of residual blocks and
strided convolutions; the decoder mirrors it with transposed convolutions.  The
two share the same ``ratios`` so the total stride matches and a round trip
preserves length (see :mod:`acoustok.modules.conv`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .activations import make_activation
from .conv import SConv1d


def _prod(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


class ResidualUnit(nn.Module):
    """Dilated residual block: two convs with a skip connection."""

    def __init__(
        self,
        dim: int,
        dilation: int,
        kernel_size: int = 3,
        activation: str = "elu",
        norm: str = "weight_norm",
        pad_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            make_activation(activation, dim),
            SConv1d(dim, dim, kernel_size, dilation=dilation, norm=norm, pad_mode=pad_mode),
            make_activation(activation, dim),
            SConv1d(dim, dim, 1, norm=norm, pad_mode=pad_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Residual units followed by a strided downsampling convolution."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int,
        dilations: Sequence[int],
        activation: str,
        norm: str,
        pad_mode: str,
    ) -> None:
        super().__init__()
        units: list[nn.Module] = [
            ResidualUnit(in_dim, d, activation=activation, norm=norm, pad_mode=pad_mode)
            for d in dilations
        ]
        units.append(make_activation(activation, in_dim))
        units.append(
            SConv1d(
                in_dim, out_dim, kernel_size=2 * stride, stride=stride, norm=norm, pad_mode=pad_mode
            )
        )
        self.block = nn.Sequential(*units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
