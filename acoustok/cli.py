"""Command-line interface for inspecting the codec and (de)tokenizing audio.

Without a checkpoint the model uses random weights, so ``encode``/``decode``
exercise the full plumbing but do not reconstruct meaningful audio.  Pass
``--checkpoint`` to load trained weights saved with :meth:`Codec.save`.
"""

from __future__ import annotations

import argparse

import torch

from ._version import __version__
from .bandwidth import bandwidth_for_num_quantizers, token_rate
from .codec import Codec
from .config import CodecConfig
from .io import load_tokens
from .tokenizer import AudioTokenizer


def _build_tokenizer(checkpoint: str | None, bandwidth: float | None) -> AudioTokenizer:
    if checkpoint:
        return AudioTokenizer.from_pretrained(checkpoint, bandwidth=bandwidth)
    return AudioTokenizer(bandwidth=bandwidth)


def _cmd_info(args: argparse.Namespace) -> int:
    config = CodecConfig()
    if args.checkpoint:
        config = Codec.from_pretrained(args.checkpoint).config
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


def _cmd_encode(args: argparse.Namespace) -> int:
    tokenizer = _build_tokenizer(args.checkpoint, args.bandwidth)
    codes = tokenizer.encode_file(args.input, n_quantizers=args.quantizers)
    tokenizer.tokens_to_file(codes, args.output)
    print(f"encoded {args.input} -> {args.output} (codes {tuple(codes.shape)})")
    return 0


def _cmd_decode(args: argparse.Namespace) -> int:
    tokenizer = _build_tokenizer(args.checkpoint, None)
    codes, _ = load_tokens(args.input)
    tokenizer.decode_to_file(codes, args.output)
    print(f"decoded {args.input} -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acoustok", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"acoustok {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="print codec configuration and bandwidth grid")
    info.add_argument("--checkpoint", help="optional trained checkpoint to inspect")
    info.set_defaults(func=_cmd_info)

    enc = sub.add_parser("encode", help="encode a WAV file to a token archive")
    enc.add_argument("input", help="input .wav file")
    enc.add_argument("output", help="output .npz token archive")
    enc.add_argument("--checkpoint", help="trained codec checkpoint")
    enc.add_argument("--bandwidth", type=float, help="target bitrate in kbps")
    enc.add_argument("--quantizers", type=int, help="explicit number of quantizers")
    enc.set_defaults(func=_cmd_encode)

    dec = sub.add_parser("decode", help="decode a token archive back to a WAV file")
    dec.add_argument("input", help="input .npz token archive")
    dec.add_argument("output", help="output .wav file")
    dec.add_argument("--checkpoint", help="trained codec checkpoint")
    dec.set_defaults(func=_cmd_decode)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
