"""Quickstart: encode a sine wave to acoustic tokens and decode it back.

Run with::

    python examples/01_quickstart.py
"""

import torch

from acoustok import AudioTokenizer
from acoustok.utils import seed_everything


def main() -> None:
    seed_everything(0)
    tokenizer = AudioTokenizer()  # 24 kHz, 8-level residual quantizer

    # One second of a 220 Hz tone.
    sr = tokenizer.sample_rate
    t = torch.arange(sr) / sr
    wav = 0.5 * torch.sin(2 * torch.pi * 220 * t)

    codes = tokenizer.encode(wav, bandwidth=6.0)
    recon = tokenizer.decode(codes)

    print(f"input samples : {wav.shape[-1]}")
    print(f"codes shape   : {tuple(codes.shape)}  (batch, n_quantizers, frames)")
    print(f"tokens/second : {tokenizer.token_rate(bandwidth=6.0):.0f}")
    print(f"recon samples : {recon.shape[-1]}")


if __name__ == "__main__":
    main()
