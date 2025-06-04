import torch

from acoustok import AudioTokenizer
from acoustok.cli import main


def test_info_runs(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "frame rate" in out
    assert "kbps" in out


def test_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "acoustok" in capsys.readouterr().out


def test_encode_then_decode_files(tmp_path):
    # Make a small WAV the CLI can read.
    tok = AudioTokenizer()
    sr = tok.sample_rate
    wav = (0.3 * torch.sin(torch.linspace(0, 50, sr))).unsqueeze(0)
    from acoustok.io import save_audio

    wav_path = tmp_path / "in.wav"
    tokens_path = tmp_path / "codes.npz"
    out_path = tmp_path / "out.wav"
    save_audio(wav_path, wav, sr)

    assert main(["encode", str(wav_path), str(tokens_path), "--quantizers", "2"]) == 0
    assert tokens_path.exists()
    assert main(["decode", str(tokens_path), str(out_path)]) == 0
    assert out_path.exists()
