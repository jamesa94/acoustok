import torch

from acoustok import functional as fn


def test_encode_decode_roundtrip(tiny_codec):
    x = torch.randn(1, 1, 64)
    codes = fn.encode(x, tiny_codec)
    wav = fn.decode(codes, tiny_codec)
    assert wav.shape[-1] == 64


def test_reconstruct_matches_input_length(tiny_codec):
    x = torch.randn(1, 1, 64)
    recon = fn.reconstruct(x, tiny_codec)
    assert recon.shape == x.shape


def test_reconstruct_respects_bandwidth(tiny_codec):
    x = torch.randn(1, 1, 64)
    recon = fn.reconstruct(x, tiny_codec, n_quantizers=1)
    assert recon.shape == x.shape
