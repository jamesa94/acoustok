"""Single-codebook vector quantization with EMA codebook updates.

The codebook learns through exponential moving averages of the encoder
statistics (the VQ-VAE-2 / SoundStream recipe) rather than by gradient descent on
the embedding table, which is markedly more stable.  Gradients still reach the
encoder through the straight-through estimator, and a small commitment loss keeps
the encoder outputs close to their assigned codes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distrib import ema_inplace, kmeans, laplace_smoothing, sample_vectors


class EuclideanCodebook(nn.Module):
    """A codebook quantizing vectors to their nearest entry under L2 distance."""

    # Declared so the type checker treats the registered buffers as plain tensors
    # rather than the ``Tensor | Module`` that ``nn.Module.__getattr__`` returns.
    inited: torch.Tensor
    cluster_size: torch.Tensor
    embed: torch.Tensor
    embed_avg: torch.Tensor

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead: float = 2.0,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead = threshold_ema_dead

        embed = torch.zeros(codebook_size, dim) if kmeans_init else torch.randn(codebook_size, dim)
        self.register_buffer("inited", torch.tensor(not kmeans_init, dtype=torch.bool))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.no_grad()
    def _init_embed(self, data: torch.Tensor) -> None:
        if bool(self.inited):
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.copy_(embed)
        self.embed_avg.copy_(embed.clone())
        self.cluster_size.copy_(cluster_size)
        self.inited.fill_(True)

    @torch.no_grad()
    def _expire_dead_codes(self, samples: torch.Tensor) -> None:
        if self.threshold_ema_dead <= 0:
            return
        dead = self.cluster_size < self.threshold_ema_dead
        if not bool(dead.any()):
            return
        replacements = sample_vectors(samples, self.codebook_size)
        self.embed[dead] = replacements[dead]
        self.cluster_size[dead] = float(self.threshold_ema_dead)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbour indices for flat ``(N, dim)`` inputs."""
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(dim=1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(dim=0, keepdim=True)
        )
        return dist.max(dim=-1).indices

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        return F.embedding(codes, self.embed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        codes = self.quantize(x.reshape(-1, shape[-1]))
        return codes.view(*shape[:-1])

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.dequantize(codes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if self.training and not bool(self.inited):
            self._init_embed(flat)

        codes = self.quantize(flat)
        quantized = self.dequantize(codes).view(*shape)
        codes = codes.view(*shape[:-1])

        if self.training:
            onehot = F.one_hot(codes.reshape(-1), self.codebook_size).type(flat.dtype)
            ema_inplace(self.cluster_size, onehot.sum(dim=0), self.decay)
            embed_sum = onehot.t() @ flat
            ema_inplace(self.embed_avg, embed_sum, self.decay)

            total = self.cluster_size.sum()
            smoothed = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon) * total
            )
            self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
            self._expire_dead_codes(flat)

        return quantized, codes


class VectorQuantizer(nn.Module):
    """Codebook + optional projections, straight-through and commitment loss.

    Operates on channel-first tensors ``(batch, dim, time)`` so it drops straight
    into the convolutional codec.  When ``codebook_dim`` differs from ``dim`` the
    quantization happens in the lower-dimensional projected space (the factorised
    codebook trick from the DAC paper), which improves codebook usage.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        decay: float = 0.99,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        threshold_ema_dead: float = 2.0,
        commitment_weight: float = 1.0,
    ) -> None:
        super().__init__()
        codebook_dim = codebook_dim or dim
        self.commitment_weight = commitment_weight
        self.codebook_size = codebook_size
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self._codebook = EuclideanCodebook(
            codebook_dim,
            codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            threshold_ema_dead=threshold_ema_dead,
        )

    @property
    def codebook(self) -> torch.Tensor:
        return self._codebook.embed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x.transpose(1, 2))
        return self._codebook.encode(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.project_out(self._codebook.decode(codes))
        return quantized.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        projected = self.project_in(x)
        quantized, codes = self._codebook(projected)

        loss = x.new_zeros(())
        if self.training:
            # Straight-through estimator: forward the codes, backward the identity.
            quantized = projected + (quantized - projected).detach()
            if self.commitment_weight > 0:
                loss = F.mse_loss(projected, quantized.detach()) * self.commitment_weight

        out = self.project_out(quantized).transpose(1, 2)
        return out, codes, loss
