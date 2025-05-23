"""Produce a flat token stream ready to feed an autoregressive audio LM.

Audio language models consume a 1-D sequence of integers.  ``flatten_codes``
interleaves the residual levels within each frame and (optionally) offsets each
level into its own slice of the vocabulary so the whole stream indexes a single
embedding table.

    python examples/04_tokens_for_lm.py
"""

import torch

from acoustok import AudioTokenizer, flatten_codes, num_tokens, unflatten_codes


def main() -> None:
    tokenizer = AudioTokenizer()
    codebook_size = tokenizer.codec.config.codebook_size

    wav = torch.randn(1, 1, tokenizer.sample_rate // 2)  # half a second
    codes = tokenizer.encode(wav, bandwidth=3.0)

    stream = flatten_codes(codes, offset=True, codebook_size=codebook_size)
    vocab = codes.shape[1] * codebook_size

    print(f"codes shape       : {tuple(codes.shape)}")
    print(f"flat stream length: {stream.shape[-1]} (== {num_tokens(codes)} tokens)")
    print(f"vocabulary size   : {vocab}")
    print(f"token id range    : [{int(stream.min())}, {int(stream.max())}]")

    # The transform is exactly invertible.
    restored = unflatten_codes(
        stream, num_quantizers=codes.shape[1], offset=True, codebook_size=codebook_size
    )
    assert torch.equal(restored, codes)
    print("round-trip flatten/unflatten: OK")


if __name__ == "__main__":
    main()
