import torch

from acoustok.modules.seanet import SEANetDecoder, SEANetEncoder


def test_encoder_downsamples_by_hop():
    enc = SEANetEncoder(dimension=16, n_filters=4, ratios=(2, 2)).eval()
    x = torch.randn(2, 1, 64)
    z = enc(x)
    assert z.shape[0] == 2
    assert z.shape[1] == 16
    assert z.shape[2] == 64 // 4  # hop = 2 * 2


def test_decoder_upsamples_by_hop():
    dec = SEANetDecoder(dimension=16, n_filters=4, ratios=(2, 2)).eval()
    z = torch.randn(2, 16, 16)
    wav = dec(z)
    assert wav.shape == (2, 1, 16 * 4)


def test_encoder_decoder_preserve_length():
    enc = SEANetEncoder(dimension=8, n_filters=4, ratios=(2, 2, 2)).eval()
    dec = SEANetDecoder(dimension=8, n_filters=4, ratios=(2, 2, 2)).eval()
    x = torch.randn(1, 1, 8 * 5)  # multiple of hop = 8
    out = dec(enc(x))
    assert out.shape[-1] == x.shape[-1]


def test_snake_activation_path_runs():
    enc = SEANetEncoder(dimension=8, n_filters=4, ratios=(2, 2), activation="snake").eval()
    z = enc(torch.randn(1, 1, 32))
    assert z.shape == (1, 8, 8)
