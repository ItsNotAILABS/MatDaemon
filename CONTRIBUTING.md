# Contributing to MatDaemon

MatDaemon is built for focused, high-impact contributions around AI matrix compute.

## Good First Contributions

- add benchmark profiles
- add backend implementations
- improve CUDA kernels
- add API examples
- add workload-specific docs for RAG, agents, or simulations
- publish hardware benchmark reports

## Development

```bash
python -m pip install -e .[dev]
pytest -q
```

## Benchmark Before Claims

When adding performance claims, include benchmark output, hardware notes, backend, dtype, and matrix shape.

## Pull Request Checklist

- tests pass
- docs updated when behavior changes
- benchmark added for performance-sensitive changes
- CPU-only install still works
- CUDA imports remain optional
