# MatDaemon

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AI Native](https://img.shields.io/badge/AI-native-black)](#ai-native-examples)
[![MCP Server](https://img.shields.io/badge/MCP-server-111827)](#mcp-server)
[![CUDA Optional](https://img.shields.io/badge/CUDA-optional-76B900)](#cuda-backend)
[![GitHub Callable](https://img.shields.io/badge/GitHub-callable-24292f)](#github-callable)

**MatDaemon is an AI-native matrix compute platform: SDK, async daemon, CLI, REST API, MCP server, GitHub Action, benchmarks, and CUDA backend surface in one lightweight repo.**

It is built for agents, RAG systems, embedding pipelines, simulations, and ML automation that need fast matrix multiplication without turning every project into a custom compute stack.

## Install

```bash
pip install matdaemon
pip install "matdaemon[api]"   # HTTP API
pip install "matdaemon[mcp]"   # MCP server
```

From source:

```bash
git clone https://github.com/ItsNotAILABS/MatDaemon.git
cd MatDaemon
python -m pip install -e .[dev,api,mcp]
pytest -q
```

## Product Surfaces

| Surface | Command / API | Use |
| --- | --- | --- |
| SDK | `md.matmul(A, B)` | direct Python integration |
| Daemon | `md.MatDaemon()` | async in-process agent jobs |
| CLI | `matdaemon matmul A.npy B.npy` | terminal workflows |
| HTTP API | `POST /v1/jobs/matmul` | mini platform jobs |
| MCP Server | `matdaemon mcp` | tool-calling AI clients |
| GitHub Action | `matdaemon-benchmark` | call MatDaemon from GitHub Actions |
| CUDA | `backend="cuda"` | CuPy RawKernel backend on GPU hosts |

## SDK Demo

```python
import numpy as np
import matdaemon as md

A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)
C = md.matmul(A, B, backend="auto")
```

## Mini Platform API

Start it:

```bash
matdaemon serve --host 0.0.0.0 --port 8000
```

Submit an async job:

```bash
curl -X POST http://localhost:8000/v1/jobs/matmul \
  -H 'content-type: application/json' \
  -d '{"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]], "backend": "numpy", "use_case": "agent-memory-routing"}'
```

Poll and fetch result:

```bash
curl http://localhost:8000/v1/jobs/<job_id>
curl http://localhost:8000/v1/jobs/<job_id>/result
```

Discover use cases:

```bash
curl http://localhost:8000/v1/use-cases
```

## MCP Server

Run it:

```bash
matdaemon mcp
```

MCP tools:

- `matdaemon_matmul`
- `matdaemon_similarity_top_k`
- `matdaemon_use_cases`

Client config shape:

```json
{
  "command": "matdaemon",
  "args": ["mcp"]
}
```

See [docs/MCP.md](docs/MCP.md).

## GitHub Callable

```yaml
- uses: ItsNotAILABS/MatDaemon/.github/actions/matdaemon-benchmark@main
  with:
    profile: ai
    backends: numpy tiled
    repetitions: "1"
    strict: "true"
```

See [docs/GITHUB_ACTION.md](docs/GITHUB_ACTION.md).

## AI-Native Examples

```bash
python examples/agent_embedding_router.py
python examples/local_rag_similarity.py
```

Use cases:

- agent memory routing
- local RAG similarity search
- embedding projection
- attention-style matrix blocks
- simulation workers
- local AI compute nodes

## Benchmarks

```bash
python benchmarks/benchmark_suite.py --quick
python benchmarks/benchmark_suite.py --profile ai --backends auto numpy tiled --output benchmarks/results-ai
python benchmarks/benchmark_suite.py --profile quick --backends numpy tiled --strict --output benchmarks/results
```

## CUDA Backend

MatDaemon preserves the specialized CUDA RawKernel backend under:

```text
backends/cuda_backend.py
```

The legacy misspelled path exists as a compatibility shim:

```text
backends/cude_backend.py
```

CPU installs stay lightweight. CUDA imports are optional and only required when `backend="cuda"` is requested.

## Backend Guide

| Backend | Use it when |
| --- | --- |
| `auto` | pick CUDA when available, otherwise route CPU/tiled by output size |
| `numpy` | direct NumPy BLAS/LAPACK path |
| `tiled` | block-wise CPU execution for large outputs |
| `cuda` | specialized CuPy RawKernel GEMM backend |

## Platform Docs

- [MCP guide](docs/MCP.md)
- [GitHub Action guide](docs/GITHUB_ACTION.md)
- [Platform guide](docs/PLATFORM.md)
- [Benchmark guide](docs/BENCHMARKING.md)
- [Launch checklist](docs/LAUNCH.md)
- [Product surface](docs/PRODUCT.md)

## License

MIT License.
