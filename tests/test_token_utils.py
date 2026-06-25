import torch

from acoustok.token_utils import flatten_codes, num_tokens, unflatten_codes


def test_flatten_unflatten_roundtrip():
    codes = torch.randint(0, 16, (2, 4, 5))
    flat = flatten_codes(codes)
    assert flat.shape == (2, 20)
    restored = unflatten_codes(flat, num_quantizers=4)
    assert torch.equal(restored, codes)


def test_flatten_is_frame_major():
    # One example, 2 levels, 2 frames; layout must be [t0q0, t0q1, t1q0, t1q1].
    codes = torch.tensor([[[1, 3], [2, 4]]])  # (1, 2, 2)
    flat = flatten_codes(codes)
    assert flat.tolist() == [[1, 2, 3, 4]]


def test_offset_shifts_each_level_into_its_own_range():
    codes = torch.randint(0, 16, (1, 3, 4))
    flat = flatten_codes(codes, offset=True, codebook_size=16)
    assert flat.max() < 3 * 16
    restored = unflatten_codes(flat, num_quantizers=3, offset=True, codebook_size=16)
    assert torch.equal(restored, codes)


def test_num_tokens_counts_levels_times_frames():
    assert num_tokens(torch.zeros(1, 4, 5)) == 20
    assert num_tokens(torch.zeros(4, 5)) == 20
