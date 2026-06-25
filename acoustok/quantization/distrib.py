"""Distribution helpers for vector-quantizer codebooks.

These are the small numerical routines a codebook needs: exponential moving
averages, Laplace smoothing of cluster counts, vector sampling and a tiny
k-means used for data-dependent initialisation.
"""

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


def kmeans(
    samples: torch.Tensor, num_clusters: int, num_iters: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's algorithm used to initialise a codebook from real data.

    Returns the cluster centroids and the number of samples assigned to each.
    Empty clusters keep their previous centroid rather than collapsing to zero.
    """
    means = sample_vectors(samples, num_clusters)
    bins = torch.zeros(num_clusters, device=samples.device)
    for _ in range(num_iters):
        diffs = (
            samples.pow(2).sum(dim=1, keepdim=True)
            - 2 * samples @ means.t()
            + means.pow(2).sum(dim=1)
        )
        buckets = (-diffs).max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters).float()
        zero_mask = bins == 0
        bins_clamped = bins.clamp(min=1)

        new_means = torch.zeros_like(means)
        new_means.index_add_(0, buckets, samples)
        new_means = new_means / bins_clamped[:, None]
        means = torch.where(zero_mask[:, None], means, new_means)
    return means, bins
