import pytest

from acoustok.bandwidth import (
    bandwidth_for_num_quantizers,
    num_quantizers_for_bandwidth,
    token_rate,
)


def test_bandwidth_and_quantizers_are_inverse():
    frame_rate, codebook = 75.0, 1024
    for n in range(1, 9):
        bw = bandwidth_for_num_quantizers(frame_rate, codebook, n)
        back = num_quantizers_for_bandwidth(frame_rate, codebook, bw, max_quantizers=8)
        assert back == n


def test_none_bandwidth_uses_all_quantizers():
    assert num_quantizers_for_bandwidth(75.0, 1024, None, max_quantizers=8) == 8
    assert num_quantizers_for_bandwidth(75.0, 1024, 0.0, max_quantizers=8) == 8


def test_bandwidth_is_clamped_to_max():
    assert num_quantizers_for_bandwidth(75.0, 1024, 1000.0, max_quantizers=8) == 8


def test_small_bandwidth_floors_to_at_least_one():
    assert num_quantizers_for_bandwidth(75.0, 1024, 0.01, max_quantizers=8) == 1


def test_token_rate_scales_with_quantizers():
    assert token_rate(75.0, 4) == pytest.approx(300.0)
