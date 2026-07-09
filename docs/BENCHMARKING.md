# Benchmarking MatDaemon

MatDaemon includes benchmark tooling for CPU, tiled CPU, auto backend selection, and CUDA hosts.

## Quick Smoke

```bash
python benchmarks/benchmark_suite.py --quick
```

## Launch Profile

```bash
python benchmarks/benchmark_suite.py --profile launch --backends numpy tiled --output benchmarks/results
```

## AI Profile

```bash
python benchmarks/benchmark_suite.py --profile ai --backends auto numpy tiled --output benchmarks/results-ai
```

The AI profile covers embedding projection, attention-style blocks, RAG similarity, and agent memory scan shapes.

## CUDA Profile

```bash
python -m pip install -e .[cuda]
python benchmarks/benchmark_suite.py --profile launch --backends numpy cuda --output benchmarks/results-cuda
```

## Custom Shapes

```bash
python benchmarks/benchmark_suite.py \
  --shape rag-large:4096x768x50000 \
  --shape attention-wide:2048x4096x2048 \
  --backends numpy tiled
```

## Outputs

When `--output` is passed, the suite writes:

- `benchmark-results.json`
- `benchmark-results.md`

Use the Markdown output for release notes, README snippets, launch posts, and benchmark issues.
