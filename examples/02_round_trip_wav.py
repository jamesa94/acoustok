"""Round-trip a WAV file through the codec and write the reconstruction.

Generates a short chirp on the fly (so the example is self-contained), tokenizes
it, then decodes back to ``reconstruction.wav``.

    python examples/02_round_trip_wav.py
"""

import tempfile
from pathlib import Path

import torch

from acoustok import AudioTokenizer
from acoustok.io import save_audio


def main() -> None:
    tokenizer = AudioTokenizer()
    sr = tokenizer.sample_rate

    # A linear chirp from 200 Hz to 2 kHz.
    n = sr  # one second
    t = torch.arange(n) / sr
    freq = torch.linspace(200, 2000, n)
    chirp = 0.4 * torch.sin(2 * torch.pi * torch.cumsum(freq, 0) / sr)

    workdir = Path(tempfile.mkdtemp())
    src = workdir / "chirp.wav"
    save_audio(src, chirp.unsqueeze(0), sr)

    codes = tokenizer.encode_file(src, bandwidth=6.0)
    tokenizer.decode_to_file(codes, "reconstruction.wav")
    print(f"wrote reconstruction.wav from {codes.shape[1]} quantizer levels")


if __name__ == "__main__":
    main()
