# MatDaemon

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AI Native](https://img.shields.io/badge/AI-native-black)](#ai-native-examples)
[![MCP Server](https://img.shields.io/badge/MCP-server-111827)](#mcp-server)
[![CUDA Optional](https://img.shields.io/badge/CUDA-optional-76B900)](#cuda-backend)
[![GitHub Callable](https://img.shields.io/badge/GitHub-callable-24292f)](#github-callable)

**MatDaemon is an AI-native matrix compute platform: SDK, async daemon, CLI, REST API, MCP server, GitHub Action, benchmarks, and CUDA backend surface in one lightweight repo.**
[![CUDA Optional](https://img.shields.io/badge/CUDA-optional-76B900)](#cuda-backend)

**MatDaemon is an AI-native matrix compute platform: SDK, async daemon, CLI, REST API, benchmarks, and CUDA backend surface in one lightweight repo.**

It is built for agents, RAG systems, embedding pipelines, simulations, and ML automation that need fast matrix multiplication without turning every project into a custom compute stack.

## Install

```bash
pip install matdaemon
pip install "matdaemon[api]"   # HTTP API
pip install "matdaemon[mcp]"   # MCP server
```

Install the API platform surface:

```bash
pip install "matdaemon[api]"
matdaemon serve --host 0.0.0.0 --port 8000
```

Install everything from source:

```bash
git clone https://github.com/ItsNotAILABS/MatDaemon.git
cd MatDaemon
python -m pip install -e .[dev,api]
pytest -q
```

Docker launch:

```bash
docker compose up --build
```

## 10-Second SDK Demo

```python
import numpy as np
import matdaemon as md

A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)
C = md.matmul(A, B, backend="auto")
```

## Mini Platform API

C = md.matmul(A, B, backend="auto")
```

## Product Surfaces

| Surface | Command / API | Use |
| --- | --- | --- |
| SDK | `md.matmul(A, B)` | direct Python integration |
| Daemon | `md.MatDaemon()` | async agent and worker jobs |
| CLI | `matdaemon matmul A.npy B.npy` | terminal workflows |
| API | `POST /v1/matmul` | local service and platform integration |
| Benchmarks | `benchmark_suite.py` | launch reports and hardware proof |
| CUDA | `backend="cuda"` | CuPy RawKernel backend on GPU hosts |

## CLI

```bash
matdaemon matmul A.npy B.npy --backend auto --output result.npy
matdaemon benchmark --size 1024 --backend tiled
matdaemon serve --host 0.0.0.0 --port 8000
```

## REST API

```bash
curl -X POST http://localhost:8000/v1/matmul \
  -H 'content-type: application/json' \
  -d '{"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]], "backend": "auto"}'
```

## AI-Native Examples

MatDaemon ships runnable examples for AI workloads:

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

Quick smoke:

```bash
python benchmarks/benchmark_suite.py --quick
```

Launch profile:

```bash
python benchmarks/benchmark_suite.py --profile launch --backends numpy tiled --output benchmarks/results
```

AI profile:

```bash
python benchmarks/benchmark_suite.py --profile ai --backends auto numpy tiled --output benchmarks/results-ai
```

CUDA profile:

```bash
python -m pip install -e .[cuda]
python benchmarks/benchmark_suite.py --profile launch --backends numpy cuda --output benchmarks/results-cuda
```

The suite emits JSON and Markdown reports so benchmark results can become release notes, launch posts, or GitHub issues.

## CUDA Backend

MatDaemon restores and preserves the specialized CUDA RawKernel backend under:

```text
backends/cuda_backend.py
```

The legacy misspelled path also exists as a compatibility shim:

```text
backends/cude_backend.py
```

CPU installs stay lightweight. CUDA imports are optional and only required when `backend="cuda"` is requested.

## Python Daemon

```python
import time
import numpy as np
import matdaemon as md

A = np.eye(512, dtype=np.float32)
B = np.ones((512, 512), dtype=np.float32)

with md.MatDaemon(backend="auto") as daemon:
    task_id = daemon.submit_matmul(A, B)
    while daemon.result(task_id) is None:
        time.sleep(0.05)
    job = daemon.result(task_id)
    print(job.output_shape, job.duration_seconds, job.backend)
```

## Backend Guide

| Backend | Use it when |
| --- | --- |
| `auto` | pick CUDA when available, otherwise route CPU/tiled by output size |
| `numpy` | direct NumPy BLAS/LAPACK path |
| `tiled` | block-wise CPU execution for large outputs |
| `cuda` | specialized CuPy RawKernel GEMM backend |

## Platform Docs

- [Platform guide](docs/PLATFORM.md)
- [Benchmark guide](docs/BENCHMARKING.md)
- [Launch checklist](docs/LAUNCH.md)
- [Product surface](docs/PRODUCT.md)

CPU installs stay lightweight. CUDA imports are optional and only required when `backend="cuda"` is requested.

- persistent job queue
- result artifact storage
- streaming progress
- cancellation
- hosted demo endpoint
- Tensor Core / FP16 / TF32 backend path
- benchmark gallery from community hardware

## License

MIT License.
