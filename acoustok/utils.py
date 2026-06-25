"""Small, dependency-light utilities shared across the package."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cpu") -> torch.device:
    """Resolve a torch device, falling back to CPU when CUDA is unavailable."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    """Number of (trainable) parameters in a module."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def pad_to_multiple(
    x: torch.Tensor, multiple: int, dim: int = -1, value: float = 0.0
) -> torch.Tensor:
    """Right-pad ``x`` along ``dim`` so its length is a multiple of ``multiple``."""
    length = x.shape[dim]
    remainder = length % multiple
    if remainder == 0:
        return x
    pad = multiple - remainder
    dim = dim % x.dim()
    pads = [0, 0] * (x.dim() - dim - 1) + [0, pad]
    return torch.nn.functional.pad(x, pads, value=value)
