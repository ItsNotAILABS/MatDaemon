# MatDaemon Launch Checklist

## Install Reality

PyPI publishing is pending. Until release upload, launch and demo from source:

```bash
git clone https://github.com/ItsNotAILABS/MatDaemon.git
cd MatDaemon
python -m pip install -e .[dev,api]
pytest -q
```

## One-Command Local Demo

```bash
python -m pip install -e .[dev,api]
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

## AI Coding Platform Demo

```bash
matdaemon mcp
```

Full MCP tools to show:

- `matdaemon_platform_manifest`
- `matdaemon_backend_status`
- `matdaemon_validate_matrices`
- `matdaemon_matmul`
- `matdaemon_similarity_top_k`
- `matdaemon_use_cases`
- `matdaemon_generate_api_payload`
- `matdaemon_generate_github_action`
- `matdaemon_smoke_benchmark`

Generic MCP config:

```json
{
  "mcpServers": {
    "matdaemon": {
      "command": "matdaemon",
      "args": ["mcp"]
    }
  }
}
```

Windows editable clone fallback:

```json
{
  "mcpServers": {
    "matdaemon": {
      "command": "python",
      "args": ["-m", "matdaemon.mcp_server"]
    }
  }
}
```

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
- self-contained MCP server with no external MCP runtime dependency
- full MCP tool suite for coding platforms and AI clients
- AI use-case registry under `/v1/use-cases`
- benchmark suite under `benchmarks/`
- GitHub-callable benchmark action
- CUDA RawKernel backend under `backends/cuda_backend.py`
- simple SDK import path: `import matdaemon as md`

## Suggested Launch Copy

MatDaemon is an AI-native matrix compute platform: SDK, daemon, CLI, HTTP API, self-contained MCP server, GitHub Action, benchmarks, and optional CUDA RawKernel backend for agents, RAG, simulations, and ML automation.

## Release Steps

1. Merge the full MCP and install hardening PR.
2. Tag a release: `v0.3.1`.
3. Publish to PyPI.
4. Run the GitHub Action benchmark profile and attach artifacts to the release.
5. Run CUDA benchmarks on a GPU machine.
6. Paste generated benchmark Markdown into release notes.
7. Update GitHub description and topics from `docs/METADATA.md`.
