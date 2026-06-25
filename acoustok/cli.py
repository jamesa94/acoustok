"""Command-line interface for inspecting the codec and (de)tokenizing audio."""

from __future__ import annotations

import argparse

from ._version import __version__
from .bandwidth import bandwidth_for_num_quantizers, token_rate
from .config import CodecConfig


def _cmd_info(args: argparse.Namespace) -> int:
    config = CodecConfig()
    print(f"acoustok {__version__}")
    print(f"  sample rate : {config.sample_rate} Hz")
    print(f"  hop length  : {config.hop_length} samples")
    print(f"  frame rate  : {config.frame_rate:.2f} Hz")
    print(f"  codebook    : {config.codebook_size} entries x {config.num_quantizers} levels")
    print(f"  max bitrate : {config.max_bandwidth:.2f} kbps")
    print("  bandwidth grid:")
    for n in range(1, config.num_quantizers + 1):
        bw = bandwidth_for_num_quantizers(config.frame_rate, config.codebook_size, n)
        rate = token_rate(config.frame_rate, n)
        print(f"    {n:>2} levels -> {bw:5.2f} kbps, {rate:7.1f} tokens/s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acoustok", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"acoustok {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="print codec configuration and bandwidth grid")
    info.set_defaults(func=_cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
