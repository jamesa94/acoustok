# Examples

Small, self-contained scripts. Each one runs on CPU in a few seconds and needs
nothing but `acoustok` installed (`pip install -e .` from the repo root).

| Script | What it shows |
| --- | --- |
| [`01_quickstart.py`](01_quickstart.py) | Encode a tone to tokens and decode it back |
| [`02_round_trip_wav.py`](02_round_trip_wav.py) | Tokenize a WAV file and write the reconstruction |
| [`03_bandwidth_sweep.py`](03_bandwidth_sweep.py) | How quantizer count trades bitrate for token rate |
| [`04_tokens_for_lm.py`](04_tokens_for_lm.py) | Flatten codes into an LM-ready token stream |

> The bundled codec has **random weights**, so reconstructions are noise until you
> train the model or load a checkpoint. The token *plumbing* is exact regardless,
> which is what these examples exercise.

```bash
python examples/01_quickstart.py
```
