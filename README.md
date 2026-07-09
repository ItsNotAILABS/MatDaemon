# MatDaemon

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AI Native](https://img.shields.io/badge/AI-native-black)](#ai-native-examples)
[![CUDA Optional](https://img.shields.io/badge/CUDA-optional-76B900)](#cuda-backend)
[![GitHub Callable](https://img.shields.io/badge/GitHub-callable-24292f)](#github-callable)

**MatDaemon is an AI-native matrix compute platform: SDK, async daemon, CLI, REST API, GitHub Action, benchmarks, and CUDA backend surface in one lightweight repo.**

It is built for agents, RAG systems, embedding pipelines, simulations, and ML automation that need fast matrix multiplication without turning every project into a custom compute stack.

## Install

```bash
pip install matdaemon
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

## Product Surfaces

| Surface | Command / API | Use |
| --- | --- | --- |
| SDK | `md.matmul(A, B)` | direct Python integration |
| Daemon | `md.MatDaemon()` | async agent and worker jobs |
| CLI | `matdaemon matmul A.npy B.npy` | terminal workflows |
| API | `POST /v1/matmul` | local service and platform integration |
| GitHub Action | `matdaemon-benchmark` | call MatDaemon from GitHub Actions |
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

## GitHub Callable

MatDaemon can be called directly from GitHub Actions.

Manual run:

1. Open **Actions -> matdaemon-benchmark**.
2. Choose `quick`, `launch`, or `ai` profile.
3. Run against `numpy`, `tiled`, `auto`, or `cuda` backends.
4. Download JSON/Markdown benchmark artifacts from the run.

Reusable action:

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

Strict CI mode:

```bash
python benchmarks/benchmark_suite.py --profile quick --backends numpy tiled --strict --output benchmarks/results
```

CUDA profile:

```bash
python -m pip install -e .[cuda]
python benchmarks/benchmark_suite.py --profile launch --backends numpy cuda --output benchmarks/results-cuda
```

The suite emits JSON and Markdown reports so benchmark results can become release notes, launch posts, GitHub artifacts, or benchmark issues.

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

- [GitHub Action guide](docs/GITHUB_ACTION.md)
- [Platform guide](docs/PLATFORM.md)
- [Benchmark guide](docs/BENCHMARKING.md)
- [Launch checklist](docs/LAUNCH.md)
- [Product surface](docs/PRODUCT.md)

## Roadmap

- persistent job queue
- result artifact storage
- streaming progress
- cancellation
- hosted demo endpoint
- Tensor Core / FP16 / TF32 backend path
- benchmark gallery from community hardware

## License

MIT License.
