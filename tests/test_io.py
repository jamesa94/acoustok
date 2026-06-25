import numpy as np
import torch

from acoustok.io import load_audio, load_tokens, save_audio, save_tokens


def test_wav_roundtrip_preserves_signal(tmp_path):
    sr = 8000
    wav = (0.4 * torch.sin(torch.linspace(0, 20, 4000))).unsqueeze(0)  # (1, T)
    path = tmp_path / "tone.wav"
    save_audio(path, wav, sr)
    loaded, loaded_sr = load_audio(path)
    assert loaded_sr == sr
    assert loaded.shape == wav.shape
    # 16-bit quantization error is bounded by one LSB.
    assert torch.max(torch.abs(loaded - wav)) < 1e-3


def test_wav_resample_on_load(tmp_path):
    wav = torch.randn(1, 4000).clamp(-1, 1)
    path = tmp_path / "noise.wav"
    save_audio(path, wav, 8000)
    loaded, sr = load_audio(path, target_sr=4000)
    assert sr == 4000
    assert loaded.shape[-1] == 2000


def test_stereo_mono_mix_on_load(tmp_path):
    wav = torch.randn(2, 1000).clamp(-1, 1)
    path = tmp_path / "stereo.wav"
    save_audio(path, wav, 8000)
    mono, _ = load_audio(path, mono=True)
    assert mono.shape[0] == 1


def test_token_roundtrip(tmp_path):
    codes = torch.randint(0, 100, (1, 4, 7))
    path = tmp_path / "codes.npz"
    save_tokens(path, codes, sample_rate=24000, frame_rate=75.0)
    loaded, meta = load_tokens(path)
    assert torch.equal(loaded, codes)
    assert meta["sample_rate"] == 24000
    assert np.isclose(meta["frame_rate"], 75.0)
