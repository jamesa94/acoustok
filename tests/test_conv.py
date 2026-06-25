import pytest
import torch

from acoustok.modules.conv import (
    SConv1d,
    SConvTranspose1d,
    get_extra_padding_for_conv1d,
    pad1d,
    unpad1d,
)


@pytest.mark.parametrize("length", [16, 17, 31, 64, 100])
@pytest.mark.parametrize("stride", [1, 2, 4])
def test_sconv_output_length_is_ceil_div(length, stride):
    conv = SConv1d(1, 3, kernel_size=2 * stride if stride > 1 else 3, stride=stride)
    x = torch.randn(2, 1, length)
    out = conv(x)
    expected = -(-length // stride)  # ceil division
    assert out.shape[-1] == expected


@pytest.mark.parametrize("stride", [2, 3, 4, 5])
def test_conv_then_transpose_restores_length(stride):
    length = stride * 7
    down = SConv1d(2, 2, kernel_size=2 * stride, stride=stride)
    up = SConvTranspose1d(2, 2, kernel_size=2 * stride, stride=stride)
    x = torch.randn(1, 2, length)
    restored = up(down(x))
    assert restored.shape[-1] == length


def test_pad_unpad_roundtrip():
    x = torch.randn(1, 1, 20)
    padded = pad1d(x, (3, 5), mode="constant")
    assert padded.shape[-1] == 28
    assert torch.equal(unpad1d(padded, (3, 5)), x)


def test_reflect_pad_handles_short_input():
    x = torch.randn(1, 1, 2)
    padded = pad1d(x, (4, 4), mode="reflect")
    assert padded.shape[-1] == 10


def test_extra_padding_is_non_negative():
    x = torch.randn(1, 1, 13)
    extra = get_extra_padding_for_conv1d(x, kernel_size=4, stride=2, padding_total=2)
    assert extra >= 0


def test_negative_padding_rejected():
    with pytest.raises(ValueError):
        pad1d(torch.zeros(1, 1, 4), (-1, 0))
    with pytest.raises(ValueError):
        unpad1d(torch.zeros(1, 1, 4), (3, 3))
