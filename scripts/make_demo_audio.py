"""Generate a short demo WAV file (a chord) for trying the CLI.

    python scripts/make_demo_audio.py demo.wav
"""

from __future__ import annotations

import argparse

import torch

from acoustok.io import save_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="write a short demo WAV file")
    parser.add_argument("output", nargs="?", default="demo.wav")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    sr = args.sample_rate
    n = int(sr * args.seconds)
    t = torch.arange(n) / sr
    # An A-major triad with a gentle fade so it does not click.
    chord = sum(torch.sin(2 * torch.pi * f * t) for f in (440.0, 554.37, 659.25))
    fade = torch.linspace(0, 1, n).clamp(max=1.0)
    wav = 0.2 * chord * fade * fade.flip(0)

    save_audio(args.output, wav.unsqueeze(0), sr)
    print(f"wrote {args.output} ({args.seconds:g}s @ {sr} Hz)")


if __name__ == "__main__":
    main()
