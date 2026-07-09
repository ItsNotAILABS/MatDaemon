# MatDaemon

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**MatDaemon is a high-performance matrix multiplication SDK and async daemon for agentic AI, ML pipelines, simulations, and automation systems.**

It gives you one clean surface for CPU NumPy, memory-aware tiled CPU execution, and optional CUDA acceleration through CuPy. Use it as a tiny SDK, a background daemon, or a CLI tool when matrix jobs need to be easy to ship, easy to benchmark, and safer around large outputs.

## Why MatDaemon

Large matrix jobs can block agent runtimes, crash small workers, or turn a simple pipeline into manual memory management. MatDaemon packages the core flow as a product-ready compute module:

- **Simple SDK:** `md.matmul(A, B)`
- **Backend selection:** `auto`, `numpy`, `tiled`, or `cuda`
- **Async daemon:** queue matrix jobs without blocking the caller
- **Memory-aware tiling:** route large outputs through tiled execution
- **Optional CUDA:** lazy CuPy backend, CPU installs remain lightweight
- **CLI:** multiply `.npy` files and run quick benchmarks
- **Proof harness:** tests and benchmarks compare correctness against NumPy

## Install

```bash
pip install matdaemon
```

For local development from the repo:

```bash
git clone https://github.com/ItsNotAILABS/MatDaemon.git
cd MatDaemon
python -m pip install -e .[dev]
pytest -q
```

Optional CUDA support requires a CuPy package that matches your CUDA runtime:

```bash
python -m pip install -e .[cuda]
# or install the exact CuPy build for your machine, for example:
pip install cupy-cuda12x
```

## SDK Quickstart

```python
import numpy as np
import matdaemon as md

A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

result = md.matmul(A, B)                  # automatic backend selection
result_cpu = md.matmul(A, B, backend="numpy")
result_tiled = md.matmul(A, B, backend="tiled")
```

## Async Daemon

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

## CLI

Multiply two `.npy` matrices:

```bash
matdaemon matmul A.npy B.npy --backend auto --output result.npy
```

Run a quick benchmark:

```bash
matdaemon benchmark --size 1024 --backend tiled
```

Run the benchmark harness:

```bash
python benchmarks/benchmark_matmul.py --sizes 256 512 1024 --backend auto
```

## Backend Guide

| Backend | Use it when |
| --- | --- |
| `auto` | You want MatDaemon to pick CUDA if available, otherwise CPU/tiled based on output size. |
| `numpy` | You want direct NumPy BLAS/LAPACK performance for normal workloads. |
| `tiled` | You want predictable block-wise CPU execution for larger outputs. |
| `cuda` | You have CuPy + CUDA installed and want GPU execution. |

CUDA is imported lazily. CPU-only installs do not break if CuPy is missing. If `backend="cuda"` is requested without a working CUDA stack, MatDaemon raises a clear `CudaUnavailableError`.

## Product Surface

MatDaemon is built to become a compute substrate for larger systems:

- agent orchestrators that need non-blocking matrix jobs
- ML and embedding pipelines that need a small GEMM utility layer
- simulation tools that need repeatable benchmark records
- local-first AI runtimes that need CPU/GPU flexibility
- future HTTP workers and operator dashboards

See [docs/PRODUCT.md](docs/PRODUCT.md) for the product surface and next gates.

## Current Status

This repo now ships the core SDK surface, CLI, tests, benchmark harness, and optional CUDA backend boundary. CUDA performance depends on local hardware, CuPy version, CUDA runtime, and matrix shape. Always benchmark on the target machine before publishing hardware-specific claims.

## Roadmap

- HTTP job service for remote matrix workers
- persistent result artifact store
- task cancellation and progress events
- Tensor Core / FP16 / TF32 backend path
- benchmark report publishing
- operator UI for queued compute jobs

## License

MIT License.
