import pytest
import torch

from acoustok.quantization import ResidualVQ


def test_codes_shape_and_range():
    rvq = ResidualVQ(num_quantizers=4, dim=8, codebook_size=16).eval()
    codes = rvq.encode(torch.randn(2, 8, 6))
    assert codes.shape == (2, 4, 6)
    assert codes.min() >= 0 and codes.max() < 16


def test_decode_matches_forward_quantized():
    rvq = ResidualVQ(num_quantizers=3, dim=8, codebook_size=16).eval()
    x = torch.randn(2, 8, 5)
    result = rvq(x)
    codes = rvq.encode(x)
    assert torch.equal(result.codes, codes)
    assert torch.allclose(rvq.decode(codes), result.quantized, atol=1e-5)


def test_fewer_quantizers_uses_prefix_of_levels():
    rvq = ResidualVQ(num_quantizers=4, dim=8, codebook_size=16).eval()
    x = torch.randn(1, 8, 4)
    codes_full = rvq.encode(x, n_quantizers=4)
    codes_two = rvq.encode(x, n_quantizers=2)
    assert codes_two.shape[1] == 2
    # The first two levels are identical regardless of how many are requested.
    assert torch.equal(codes_two, codes_full[:, :2])


def test_more_levels_reduce_reconstruction_error():
    rvq = ResidualVQ(num_quantizers=4, dim=16, codebook_size=64).eval()
    x = torch.randn(4, 16, 20)
    err_one = (x - rvq.decode(rvq.encode(x, n_quantizers=1))).pow(2).mean()
    err_all = (x - rvq.decode(rvq.encode(x, n_quantizers=4))).pow(2).mean()
    assert err_all <= err_one


def test_invalid_quantizer_count_raises():
    rvq = ResidualVQ(num_quantizers=2, dim=4, codebook_size=8).eval()
    with pytest.raises(ValueError):
        rvq.encode(torch.randn(1, 4, 3), n_quantizers=5)


def test_training_returns_finite_loss():
    rvq = ResidualVQ(num_quantizers=3, dim=8, codebook_size=16).train()
    # A real encoder output carries a gradient, so the commitment loss should too.
    result = rvq(torch.randn(2, 8, 6, requires_grad=True))
    assert result.loss.requires_grad
    assert torch.isfinite(result.loss)
    assert result.metrics["num_quantizers"] == 3
