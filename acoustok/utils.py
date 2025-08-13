"""Small, dependency-light utilities shared across the package."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch so a run is reproducible.

    Offline-by-default reproducibility is a core goal of the library, so this is
    used in examples and tests rather than left to the caller.
    """
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
    # F.pad pads from the last dimension backwards; build the spec accordingly.
    dim = dim % x.dim()
    pads = [0, 0] * (x.dim() - dim - 1) + [0, pad]
    return F.pad(x, pads, value=value)


def resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample audio with linear interpolation.

    This is intentionally simple — no anti-aliasing filter — and is meant for
    examples and lightweight preprocessing.  For research-grade resampling reach
    for ``torchaudio`` or ``soxr``.  ``wav`` is shaped ``(channels, time)``.
    """
    if orig_sr == target_sr:
        return wav
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    squeeze = False
    if wav.dim() == 1:
        wav = wav[None]
        squeeze = True
    new_length = int(round(wav.shape[-1] * target_sr / orig_sr))
    out = F.interpolate(wav[None].float(), size=new_length, mode="linear", align_corners=False)[0]
    return out[0] if squeeze else out


def dbfs(wav: torch.Tensor, eps: float = 1e-8) -> float:
    """Root-mean-square level of a waveform in dBFS (handy for debugging)."""
    rms = wav.float().pow(2).mean().sqrt().item()
    return 20.0 * np.log10(max(rms, eps))


def chunk_signal(wav: torch.Tensor, chunk: int, hop: int | None = None) -> torch.Tensor:
    """Split the last dim into overlapping frames of length ``chunk``."""
    hop = hop or chunk
    return wav.unfold(-1, chunk, hop)
