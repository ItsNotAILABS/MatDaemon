# MatDaemon Launch Checklist

## One-Command Local Demo

```bash
python -m pip install -e .[dev,api,mcp]
pytest -q
matdaemon platform
matdaemon benchmark --size 1024 --backend auto
matdaemon serve --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl http://localhost:8000/v1/platform
curl http://localhost:8000/v1/use-cases
```

## Docker Demo

```bash
docker compose up --build
curl http://localhost:8000/v1/platform
```

## AI Demo

```bash
matdaemon mcp
```

MCP tools to show:

- `matdaemon_matmul`
- `matdaemon_similarity_top_k`
- `matdaemon_use_cases`
- `matdaemon_platform_manifest`

## Benchmark Proof

```bash
python benchmarks/benchmark_suite.py --quick
python benchmarks/benchmark_suite.py --profile ai --backends auto numpy tiled --output benchmarks/results-ai
```

From GitHub, run **Actions -> matdaemon-benchmark -> Run workflow** and download the JSON/Markdown artifacts.

## Star Hooks

- visible platform architecture graphic in README
- one command platform contract: `matdaemon platform`
- HTTP manifest endpoint: `GET /v1/platform`
- MCP manifest tool: `matdaemon_platform_manifest`
- AI use-case registry under `/v1/use-cases`
- benchmark suite under `benchmarks/`
- GitHub-callable benchmark action
- CUDA RawKernel backend under `backends/cuda_backend.py`
- simple SDK import path: `import matdaemon as md`

## Suggested Launch Copy

MatDaemon is an AI-native matrix compute platform: SDK, daemon, CLI, HTTP API, MCP server, GitHub Action, benchmarks, and optional CUDA RawKernel backend for agents, RAG, simulations, and ML automation.

## Release Steps

1. Merge the production platform polish PR.
2. Tag a release: `v0.3.0`.
3. Publish to PyPI.
4. Run the GitHub Action benchmark profile and attach artifacts to the release.
5. Run CUDA benchmarks on a GPU machine.
6. Paste generated benchmark Markdown into release notes.
7. Update GitHub description and topics from `docs/METADATA.md`.
