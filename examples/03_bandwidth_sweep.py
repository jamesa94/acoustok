"""Show how bandwidth (number of quantizers) trades off against token rate.

    python examples/03_bandwidth_sweep.py
"""

import torch

from acoustok import AudioTokenizer


def main() -> None:
    tokenizer = AudioTokenizer()
    wav = torch.randn(1, 1, tokenizer.sample_rate)  # one second

    print(f"{'levels':>6} {'kbps':>7} {'tokens/s':>10} {'codes':>16}")
    for n in range(1, tokenizer.max_quantizers + 1):
        codes = tokenizer.encode(wav, n_quantizers=n)
        kbps = tokenizer.bandwidth_for(n)
        rate = tokenizer.token_rate(n_quantizers=n)
        print(f"{n:>6} {kbps:>7.2f} {rate:>10.0f} {str(tuple(codes.shape)):>16}")


if __name__ == "__main__":
    main()
