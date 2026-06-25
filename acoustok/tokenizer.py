"""High-level tokenizer wrapping a :class:`~acoustok.codec.Codec`.

This is the surface most users touch: give it a waveform (any of a 1-D array, a
``(channels, time)`` array or a batched ``(batch, channels, time)`` tensor) and
get back integer tokens, and vice versa.  Shape juggling, device placement and
mono down-mixing all live here so the codec stays a clean ``nn.Module``.
"""

from __future__ import annotations

import numpy as np
import torch

from .codec import Codec
from .config import CodecConfig

WaveformLike = np.ndarray | torch.Tensor


class AudioTokenizer:
    """Turn audio into acoustic tokens and back for audio language models."""

    def __init__(
        self,
        codec: Codec | None = None,
        config: CodecConfig | None = None,
        device: str = "cpu",
    ) -> None:
        if codec is None:
            codec = Codec(config)
        self.codec = codec.to(device).eval()
        self.device = torch.device(device)

    @property
    def sample_rate(self) -> int:
        return self.codec.config.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.codec.frame_rate

    @property
    def max_quantizers(self) -> int:
        return self.codec.num_quantizers

    def _prepare(self, wav: WaveformLike) -> torch.Tensor:
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        wav = wav.to(torch.float32)
        if wav.dim() == 1:
            wav = wav[None, None]
        elif wav.dim() == 2:  # (channels, time) -> add batch
            wav = wav[None]
        elif wav.dim() != 3:
            raise ValueError(f"expected 1-D, 2-D or 3-D audio, got {wav.dim()}-D")
        wav = wav.to(self.device)
        if wav.shape[1] != self.codec.config.channels:
            wav = wav.mean(dim=1, keepdim=True)
        return wav

    @torch.no_grad()
    def encode(self, wav: WaveformLike, n_quantizers: int | None = None) -> torch.Tensor:
        """Encode audio to codes ``(batch, n_quantizers, frames)``."""
        prepared = self._prepare(wav)
        return self.codec.encode(prepared, n_quantizers=n_quantizers)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes back to a waveform ``(batch, channels, time)``."""
        codes = codes.to(self.device)
        if codes.dim() == 2:
            codes = codes[None]
        return self.codec.decode(codes)
