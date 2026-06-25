import numpy as np
import torch

from acoustok import AudioTokenizer


def test_encode_accepts_various_shapes(tiny_codec, sine_wave):
    tok = AudioTokenizer(codec=tiny_codec)
    one_d = tok.encode(sine_wave)
    two_d = tok.encode(sine_wave[None])  # (1, T)
    three_d = tok.encode(sine_wave[None, None])  # (1, 1, T)
    assert one_d.shape == two_d.shape == three_d.shape
    assert one_d.dim() == 3


def test_encode_accepts_numpy(tiny_codec, sine_wave):
    tok = AudioTokenizer(codec=tiny_codec)
    codes = tok.encode(sine_wave.numpy())
    assert codes.shape[0] == 1


def test_decode_handles_unbatched_codes(tiny_codec, sine_wave):
    tok = AudioTokenizer(codec=tiny_codec)
    codes = tok.encode(sine_wave)
    wav = tok.decode(codes[0])  # drop batch dim
    assert wav.shape[0] == 1


def test_stereo_is_downmixed(tiny_codec):
    tok = AudioTokenizer(codec=tiny_codec)
    stereo = torch.randn(2, 64)  # (channels, time)
    codes = tok.encode(stereo)
    assert codes.shape[0] == 1


def test_token_rate_and_bandwidth(tiny_codec):
    tok = AudioTokenizer(codec=tiny_codec)
    rate = tok.token_rate(n_quantizers=2)
    assert rate == tok.frame_rate * 2
    assert tok.bandwidth_for(2) > 0


def test_file_roundtrip(tiny_codec, sine_wave, tmp_path):
    tok = AudioTokenizer(codec=tiny_codec)
    codes = tok.encode(sine_wave)
    token_path = tmp_path / "tokens.npz"
    wav_path = tmp_path / "out.wav"
    tok.tokens_to_file(codes, token_path)
    tok.decode_to_file(codes, wav_path)
    assert token_path.exists()
    assert wav_path.exists()
    reloaded = np.load(token_path)
    assert reloaded["codes"].shape == tuple(codes.shape)
