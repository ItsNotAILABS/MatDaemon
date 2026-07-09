# MatDaemon Launch Checklist

## One-Command Local Demo

```bash
python -m pip install -e .[dev,api]
pytest -q
matdaemon benchmark --size 1024 --backend auto
matdaemon serve --host 0.0.0.0 --port 8000
```

## Docker Demo

```bash
docker compose up --build
```

## Star Hooks

- AI-native examples under `examples/`
- benchmark suite under `benchmarks/`
- API platform surface under `matdaemon/api.py`
- CUDA RawKernel backend under `backends/cuda_backend.py`
- simple SDK import path: `import matdaemon as md`

## Suggested Launch Copy

MatDaemon is an AI-native matrix compute SDK and daemon for agents, RAG systems, simulations, and ML pipelines. It gives you NumPy, memory-aware tiling, optional CUDA, a CLI, benchmarks, and a FastAPI surface in one lightweight package.

## Release Steps

1. Merge the launch polish PR.
2. Tag a release: `v0.2.0`.
3. Publish to PyPI.
4. Run `make benchmark-ai` on CPU hardware.
5. Run CUDA benchmarks on a GPU machine.
6. Paste generated benchmark Markdown into the release notes.
