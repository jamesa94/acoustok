"""Vector quantization: single codebooks and residual stacks."""

from .residual_vq import QuantizeResult, ResidualVQ
from .vq import EuclideanCodebook, VectorQuantizer

__all__ = [
    "EuclideanCodebook",
    "VectorQuantizer",
    "ResidualVQ",
    "QuantizeResult",
]
