# acoustok

Neural audio codec and tokenizer for audio language models.

`acoustok` turns raw waveforms into compact sequences of discrete acoustic tokens
with a convolutional encoder and residual vector quantization, and reconstructs
audio from those tokens with a matching decoder.

## Install

```bash
pip install acoustok
```

## Quickstart

```python
import torch
from acoustok import AudioTokenizer

tok = AudioTokenizer()                    # 24 kHz, 8-level residual quantizer
wav = torch.randn(1, 1, 24000)            # one second of audio
codes = tok.encode(wav, bandwidth=6.0)    # (batch, n_quantizers, frames)
recon = tok.decode(codes)                 # back to a waveform
```

The bundled architecture ships with random weights — train it, or load a
checkpoint, before expecting high-fidelity reconstruction.

## License

MIT
