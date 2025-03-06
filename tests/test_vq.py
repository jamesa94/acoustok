import torch

from acoustok.quantization import EuclideanCodebook, VectorQuantizer


def test_quantized_output_is_a_codebook_entry():
    vq = VectorQuantizer(dim=8, codebook_size=32).eval()
    x = torch.randn(2, 8, 5)
    out, codes, _ = vq(x)
    # Each output column must equal the embedding indexed by its code.
    expected = torch.nn.functional.embedding(codes, vq.codebook).transpose(1, 2)
    assert torch.allclose(out, expected, atol=1e-5)


def test_codes_are_in_range():
    vq = VectorQuantizer(dim=8, codebook_size=16).eval()
    _, codes, _ = vq(torch.randn(3, 8, 7))
    assert codes.min() >= 0
    assert codes.max() < 16
    assert codes.dtype == torch.long


def test_encode_decode_matches_forward():
    vq = VectorQuantizer(dim=8, codebook_size=16).eval()
    x = torch.randn(2, 8, 6)
    forward_out, _, _ = vq(x)
    codes = vq.encode(x)
    assert torch.allclose(vq.decode(codes), forward_out, atol=1e-6)


def test_straight_through_passes_gradient():
    vq = VectorQuantizer(dim=4, codebook_size=8).train()
    x = torch.randn(1, 4, 3, requires_grad=True)
    out, _, loss = vq(x)
    (out.sum() + loss).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_ema_moves_codebook_toward_data():
    torch.manual_seed(0)
    codebook = EuclideanCodebook(dim=4, codebook_size=8, decay=0.5, threshold_ema_dead=0.0)
    codebook.train()
    data = torch.randn(256, 4)
    before = codebook.embed.clone()
    for _ in range(20):
        codebook(data)
    # The codebook should have moved (EMA update) and stay finite.
    assert not torch.allclose(before, codebook.embed)
    assert torch.isfinite(codebook.embed).all()


def test_factorised_codebook_projection():
    vq = VectorQuantizer(dim=8, codebook_size=16, codebook_dim=4).eval()
    out, codes, _ = vq(torch.randn(1, 8, 5))
    assert out.shape == (1, 8, 5)
    assert codes.shape == (1, 5)
