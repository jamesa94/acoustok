# Benchmarks

Rough timing scripts, not a rigorous benchmark suite. They are handy for spotting
large performance regressions during development.

| Script | Measures |
| --- | --- |
| [`bench_quantizer.py`](bench_quantizer.py) | Residual VQ encode+decode throughput vs. depth |

```bash
python benchmarks/bench_quantizer.py
```

Numbers are CPU, single process, and will vary machine to machine — compare
relative changes, not absolute values.
