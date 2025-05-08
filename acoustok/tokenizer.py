"""High-level tokenizer wrapping a :class:`~acoustok.codec.Codec`.

This is the surface most users touch: give it a waveform (any of a 1-D array, a
``(channels, time)`` array or a batched ``(batch, channels, time)`` tensor) and
get back integer tokens, and vice versa.  Shape juggling, device placement and
mono down-mixing all live here so the codec stays a clean ``nn.Module``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .bandwidth import bandwidth_for_num_quantizers, token_rate
from .codec import Codec
from .config import CodecConfig
from .io import load_audio, save_audio, save_tokens
from .utils import resample

PathLike = str | Path
WaveformLike = np.ndarray | torch.Tensor


class AudioTokenizer:
    """Turn audio into acoustic tokens and back for audio language models."""

    def __init__(
        self,
        codec: Codec | None = None,
        config: CodecConfig | None = None,
        device: str = "cpu",
        bandwidth: float | None = None,
    ) -> None:
        if codec is None:
            codec = Codec(config)
        self.codec = codec.to(device).eval()
        self.device = torch.device(device)
        self.bandwidth = bandwidth

    # -- convenience constructors ------------------------------------------------
    @classmethod
    def from_pretrained(
        cls, path: PathLike, device: str = "cpu", **kwargs: object
    ) -> AudioTokenizer:
        codec = Codec.from_pretrained(str(path), map_location=device)
        return cls(codec=codec, device=device, **kwargs)  # type: ignore[arg-type]

    # -- metadata ----------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self.codec.config.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.codec.frame_rate

    @property
    def max_quantizers(self) -> int:
        return self.codec.num_quantizers

    def token_rate(self, bandwidth: float | None = None, n_quantizers: int | None = None) -> float:
        n = (
            n_quantizers
            if n_quantizers is not None
            else self.codec.num_quantizers_for_bandwidth(
                bandwidth if bandwidth is not None else self.bandwidth
            )
        )
        return token_rate(self.frame_rate, n)

    def bandwidth_for(self, n_quantizers: int) -> float:
        return bandwidth_for_num_quantizers(
            self.frame_rate, self.codec.config.codebook_size, n_quantizers
        )

    # -- core API ----------------------------------------------------------------
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
    def encode(
        self,
        wav: WaveformLike,
        bandwidth: float | None = None,
        n_quantizers: int | None = None,
    ) -> torch.Tensor:
        """Encode audio to codes ``(batch, n_quantizers, frames)``."""
        prepared = self._prepare(wav)
        bw = bandwidth if bandwidth is not None else self.bandwidth
        return self.codec.encode(prepared, bandwidth=bw, n_quantizers=n_quantizers)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes back to a waveform ``(batch, channels, time)``."""
        codes = codes.to(self.device)
        if codes.dim() == 2:
            codes = codes[None]
        return self.codec.decode(codes)

    # -- file helpers ------------------------------------------------------------
    def encode_file(
        self,
        path: PathLike,
        bandwidth: float | None = None,
        n_quantizers: int | None = None,
    ) -> torch.Tensor:
        wav, sr = load_audio(path, mono=self.codec.config.channels == 1)
        if sr != self.sample_rate:
            wav = resample(wav, sr, self.sample_rate)
        return self.encode(wav, bandwidth=bandwidth, n_quantizers=n_quantizers)

    def decode_to_file(self, codes: torch.Tensor, path: PathLike) -> None:
        wav = self.decode(codes)[0]
        save_audio(path, wav, self.sample_rate)

    def tokens_to_file(self, codes: torch.Tensor, path: PathLike) -> None:
        save_tokens(
            path,
            codes,
            sample_rate=self.sample_rate,
            frame_rate=self.frame_rate,
            num_quantizers=int(codes.shape[-2]) if codes.dim() >= 2 else 1,
        )
