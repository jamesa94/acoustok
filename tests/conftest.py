"""Shared pytest fixtures: a tiny deterministic codec and synthetic audio."""

from __future__ import annotations

import math

import pytest
import torch

from acoustok import Codec, CodecConfig
from acoustok.utils import seed_everything


@pytest.fixture(autouse=True)
def _determinism() -> None:
    # Every test starts from the same RNG state so codebooks and weights match.
    seed_everything(0)


@pytest.fixture
def tiny_config() -> CodecConfig:
    """A small config that keeps tensors cheap while exercising every path."""
    return CodecConfig(
        sample_rate=1600,
        dimension=16,
        n_filters=4,
        ratios=(2, 2),
        codebook_size=64,
        num_quantizers=4,
        kernel_size=3,
        last_kernel_size=3,
        dilations=(1, 3),
    )


@pytest.fixture
def tiny_codec(tiny_config: CodecConfig) -> Codec:
    return Codec(tiny_config).eval()


@pytest.fixture
def sine_wave() -> torch.Tensor:
    """A 1-D 440 Hz sine at the tiny-config sample rate, length a hop multiple."""
    sr, freq, length = 1600, 110.0, 1600
    t = torch.arange(length, dtype=torch.float32) / sr
    return 0.5 * torch.sin(2 * math.pi * freq * t)
