"""Length-preserving 1-D convolution primitives.

A stack of strided convolutions in the encoder must be invertible in length by
the matching transposed convolutions in the decoder, otherwise the reconstructed
waveform drifts out of alignment with the input.  The padding arithmetic below
follows the SEANet / EnCodec recipe: every :class:`SConv1d` pads its input so the
output is exactly ``ceil(length / stride)`` frames, and every
:class:`SConvTranspose1d` trims the symmetric amount back off.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch >= 1.12 exposes the parametrization-based API
    from torch.nn.utils.parametrizations import weight_norm as _weight_norm
except ImportError:  # pragma: no cover - very old torch
    from torch.nn.utils import weight_norm as _weight_norm


def apply_norm(module: nn.Module, norm: str) -> nn.Module:
    if norm == "weight_norm":
        return _weight_norm(module)
    if norm == "none":
        return module
    raise ValueError(f"unknown norm {norm!r}")


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """Extra right padding so the kernel sees every frame (no dropped tail)."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """Pad the last dimension."""
    left, right = paddings
    if left < 0 or right < 0:
        raise ValueError(f"padding must be non-negative, got {paddings}")
    return F.pad(x, paddings, mode=mode, value=value)


def unpad1d(x: torch.Tensor, paddings: tuple[int, int]) -> torch.Tensor:
    """Remove ``(left, right)`` samples from the last dimension."""
    left, right = paddings
    if left < 0 or right < 0:
        raise ValueError(f"padding must be non-negative, got {paddings}")
    if left + right > x.shape[-1]:
        raise ValueError("trying to unpad more than the tensor length")
    end = x.shape[-1] - right
    return x[..., left:end]


class NormConv1d(nn.Module):
    """``nn.Conv1d`` with an optional weight normalization wrapper."""

    def __init__(self, *args: Any, norm: str = "weight_norm", **kwargs: Any) -> None:
        super().__init__()
        self.conv = apply_norm(nn.Conv1d(*args, **kwargs), norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NormConvTranspose1d(nn.Module):
    """``nn.ConvTranspose1d`` with an optional weight normalization wrapper."""

    def __init__(self, *args: Any, norm: str = "weight_norm", **kwargs: Any) -> None:
        super().__init__()
        self.convtr = apply_norm(nn.ConvTranspose1d(*args, **kwargs), norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convtr(x)


class SConv1d(nn.Module):
    """Strided conv that pads its input to keep ``ceil(length / stride)`` frames."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: str = "weight_norm",
        pad_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            norm=norm,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = (self.kernel_size - 1) * self.dilation + 1
        padding_total = kernel - self.stride
        extra = get_extra_padding_for_conv1d(x, kernel, self.stride, padding_total)
        right = padding_total // 2
        left = padding_total - right
        x = pad1d(x, (left, right + extra), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """Transposed conv that upsamples by exactly ``stride`` and trims padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm: str = "weight_norm",
    ) -> None:
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, norm=norm
        )
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convtr(x)
        padding_total = self.kernel_size - self.stride
        right = padding_total // 2
        left = padding_total - right
        return unpad1d(out, (left, right))
