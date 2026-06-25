"""Audio and token file I/O using only the standard library plus NumPy.

WAV reading/writing goes through :mod:`wave` so there is no system dependency on
libsndfile — in keeping with the offline-by-default goal.  Tokens are stored as a
compressed ``.npz`` archive carrying the codes alongside a little metadata.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import resample

PathLike = str | Path

_DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32}
_SCALES = {1: 128.0, 2: 32768.0, 4: 2147483648.0}


def load_audio(
    path: PathLike,
    target_sr: int | None = None,
    mono: bool = False,
) -> tuple[torch.Tensor, int]:
    """Load a WAV file as a float tensor in ``[-1, 1]`` shaped ``(channels, time)``."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sampwidth not in _DTYPES:
        raise ValueError(f"unsupported sample width: {sampwidth} bytes")

    data = np.frombuffer(raw, dtype=_DTYPES[sampwidth]).astype(np.float32)
    if sampwidth == 1:  # 8-bit PCM is unsigned, centred at 128
        data = (data - 128.0) / _SCALES[1]
    else:
        data = data / _SCALES[sampwidth]
    data = data.reshape(-1, n_channels).T  # (channels, time)

    wav = torch.from_numpy(np.ascontiguousarray(data))
    if mono and n_channels > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr is not None and target_sr != sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def save_audio(path: PathLike, wav: torch.Tensor, sample_rate: int) -> None:
    """Write a float waveform ``(channels, time)`` as 16-bit PCM WAV."""
    if isinstance(wav, torch.Tensor):
        array = wav.detach().cpu().float().numpy()
    else:
        array = np.asarray(wav, dtype=np.float32)
    if array.ndim == 1:
        array = array[None]
    n_channels = array.shape[0]

    clipped = np.clip(array, -1.0, 1.0)
    pcm = (clipped * (_SCALES[2] - 1)).astype(np.int16)
    interleaved = pcm.T.reshape(-1)  # (time, channels) -> flat

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(interleaved.tobytes())


def save_tokens(path: PathLike, codes: torch.Tensor, **meta: Any) -> None:
    """Save discrete codes plus metadata to a compressed ``.npz`` archive."""
    array = codes.detach().cpu().numpy().astype(np.int64)
    np.savez_compressed(path, codes=array, **meta)


def load_tokens(path: PathLike) -> tuple[torch.Tensor, dict[str, Any]]:
    """Inverse of :func:`save_tokens`; returns ``(codes, metadata)``."""
    with np.load(path, allow_pickle=False) as archive:
        codes = torch.from_numpy(archive["codes"])
        meta: dict[str, Any] = {}
        for key in archive.files:
            if key == "codes":
                continue
            value = archive[key]
            meta[key] = value.item() if value.ndim == 0 else value
    return codes, meta
