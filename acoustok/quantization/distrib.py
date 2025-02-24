"""Distribution helpers for vector-quantizer codebooks."""

from __future__ import annotations

import torch


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    """In-place ``moving_avg = decay * moving_avg + (1 - decay) * new``."""
    moving_avg.mul_(decay).add_(new, alpha=1.0 - decay)


def laplace_smoothing(x: torch.Tensor, n_categories: int, epsilon: float = 1e-5) -> torch.Tensor:
    """Additive smoothing so empty clusters never divide by zero."""
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    """Draw ``num`` rows from ``samples`` (with replacement if it is too small)."""
    n_samples, device = samples.shape[0], samples.device
    if n_samples >= num:
        indices = torch.randperm(n_samples, device=device)[:num]
    else:
        indices = torch.randint(0, n_samples, (num,), device=device)
    return samples[indices]
