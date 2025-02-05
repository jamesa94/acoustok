import torch

from acoustok.utils import (
    chunk_signal,
    count_parameters,
    pad_to_multiple,
    resample,
    seed_everything,
)


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_pad_to_multiple_lengths():
    x = torch.randn(1, 1, 13)
    out = pad_to_multiple(x, multiple=8)
    assert out.shape[-1] == 16
    # Already a multiple -> unchanged.
    assert torch.equal(pad_to_multiple(out, 8), out)


def test_resample_changes_length():
    wav = torch.randn(1, 1000)
    up = resample(wav, 8000, 16000)
    assert up.shape[-1] == 2000
    same = resample(wav, 8000, 8000)
    assert torch.equal(same, wav)


def test_count_parameters():
    layer = torch.nn.Linear(4, 3)
    assert count_parameters(layer) == 4 * 3 + 3


def test_chunk_signal_frames():
    x = torch.arange(10).float()
    frames = chunk_signal(x, chunk=4, hop=2)
    assert frames.shape[-1] == 4
