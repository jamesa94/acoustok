"""Micro-benchmark for the residual vector quantizer.

Measures encode/decode throughput at a few quantizer depths. This is a rough,
single-process timing — not a rigorous benchmark — but it is enough to catch large
regressions.

    python benchmarks/bench_quantizer.py
"""

from __future__ import annotations

import time

import torch

from acoustok.quantization import ResidualVQ


def _bench(n_quantizers: int, *, dim: int = 128, frames: int = 750, iters: int = 20) -> float:
    rvq = ResidualVQ(num_quantizers=n_quantizers, dim=dim, codebook_size=1024).eval()
    x = torch.randn(4, dim, frames)  # ~10 s of audio at 75 Hz
    with torch.no_grad():
        rvq.encode(x)  # warm-up
        start = time.perf_counter()
        for _ in range(iters):
            codes = rvq.encode(x)
            rvq.decode(codes)
        elapsed = time.perf_counter() - start
    return elapsed / iters * 1000.0  # ms / iter


def main() -> None:
    torch.manual_seed(0)
    print(f"{'levels':>6} {'ms/iter':>10}")
    for n in (1, 2, 4, 8):
        print(f"{n:>6} {_bench(n):>10.2f}")


if __name__ == "__main__":
    main()
