"""acoustok — a neural audio codec and tokenizer for audio language models.

The public surface is intentionally small:

* :class:`AudioTokenizer` — high-level encode/decode of audio to acoustic tokens.
* :class:`Codec` — the underlying encoder + residual quantizer + decoder module.
* :class:`ResidualVQ` / :class:`VectorQuantizer` — the quantization building blocks.
* :class:`CodecConfig` — the architecture description.
* file and token helpers in :mod:`acoustok.io` and :mod:`acoustok.token_utils`.
"""

from ._version import __version__
from .bandwidth import (
    bandwidth_for_num_quantizers,
    num_quantizers_for_bandwidth,
    token_rate,
)
from .codec import Codec
from .config import CodecConfig
from .functional import decode, encode, reconstruct
from .io import load_audio, load_tokens, save_audio, save_tokens
from .quantization import EuclideanCodebook, QuantizeResult, ResidualVQ, VectorQuantizer
from .token_utils import flatten_codes, num_tokens, unflatten_codes
from .tokenizer import AudioTokenizer

__all__ = [
    "__version__",
    "AudioTokenizer",
    "Codec",
    "CodecConfig",
    "ResidualVQ",
    "VectorQuantizer",
    "EuclideanCodebook",
    "QuantizeResult",
    "encode",
    "decode",
    "reconstruct",
    "load_audio",
    "save_audio",
    "load_tokens",
    "save_tokens",
    "flatten_codes",
    "unflatten_codes",
    "num_tokens",
    "num_quantizers_for_bandwidth",
    "bandwidth_for_num_quantizers",
    "token_rate",
]
