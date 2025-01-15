import math

import pytest

from acoustok import CodecConfig


def test_defaults_are_sensible():
    cfg = CodecConfig()
    assert cfg.sample_rate == 24000
    assert cfg.hop_length == 8 * 5 * 4 * 2
    assert cfg.frame_rate == pytest.approx(75.0)
    assert cfg.bits_per_codebook == pytest.approx(math.log2(1024))


def test_max_bandwidth_matches_grid():
    cfg = CodecConfig()
    # 75 Hz * 8 levels * 10 bits = 6000 bps = 6 kbps
    assert cfg.max_bandwidth == pytest.approx(6.0)


def test_sequence_fields_normalised_to_tuples():
    cfg = CodecConfig(ratios=[4, 4], dilations=[1, 2])
    assert isinstance(cfg.ratios, tuple)
    assert isinstance(cfg.dilations, tuple)
    assert cfg.hop_length == 16


def test_roundtrip_through_dict():
    cfg = CodecConfig(sample_rate=16000, ratios=(8, 5, 4, 2))
    restored = CodecConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_from_dict_ignores_unknown_keys():
    cfg = CodecConfig.from_dict({"sample_rate": 8000, "bogus": 123})
    assert cfg.sample_rate == 8000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"channels": 0},
        {"ratios": ()},
        {"ratios": (2, -1)},
        {"codebook_size": 1},
        {"num_quantizers": 0},
    ],
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        CodecConfig(**kwargs)
