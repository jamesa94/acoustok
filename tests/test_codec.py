import torch

from acoustok import Codec


def test_encode_returns_codes(tiny_codec):
    codes = tiny_codec.encode(torch.randn(2, 1, 64))
    assert codes.shape[0] == 2
    assert codes.shape[1] == tiny_codec.num_quantizers
    assert codes.shape[2] == 64 // tiny_codec.hop_length


def test_forward_reconstruction_matches_input_length(tiny_codec):
    x = torch.randn(1, 1, 70)  # not a multiple of hop on purpose
    recon, codes, loss = tiny_codec(x)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_round_trip_codes_decode_to_padded_length(tiny_codec):
    x = torch.randn(1, 1, 64)
    codes = tiny_codec.encode(x)
    wav = tiny_codec.decode(codes)
    assert wav.shape[-1] == 64


def test_bandwidth_controls_quantizer_count(tiny_config):
    codec = Codec(tiny_config).eval()
    low = codec.encode(torch.randn(1, 1, 64), bandwidth=codec.config.max_bandwidth / 4)
    high = codec.encode(torch.randn(1, 1, 64))
    assert low.shape[1] < high.shape[1]


def test_eval_mode_is_deterministic(tiny_codec):
    x = torch.randn(1, 1, 64)
    a = tiny_codec.encode(x)
    b = tiny_codec.encode(x)
    assert torch.equal(a, b)


def test_save_and_load_roundtrip(tiny_codec, tmp_path):
    path = tmp_path / "codec.pt"
    tiny_codec.save(str(path))
    loaded = Codec.from_pretrained(str(path))
    x = torch.randn(1, 1, 64)
    assert torch.equal(tiny_codec.encode(x), loaded.encode(x))
    assert loaded.config == tiny_codec.config
