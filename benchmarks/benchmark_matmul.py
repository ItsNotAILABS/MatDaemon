"""Small benchmark harness for MatDaemon marketing/proof runs.

Usage:
    python benchmarks/benchmark_matmul.py --sizes 256 512 1024 --backend auto
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import matdaemon as md


def run_case(size: int, backend: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((size, size), dtype=np.float32)
    B = rng.standard_normal((size, size), dtype=np.float32)
    start = time.perf_counter()
    result = md.matmul(A, B, backend=backend)
    duration = time.perf_counter() - start
    baseline = np.matmul(A, B)
    max_abs_error = float(np.max(np.abs(result - baseline)))
    return {
        "size": size,
        "backend": backend,
        "duration_seconds": round(duration, 6),
        "max_abs_error_vs_numpy": max_abs_error,
        "output_shape": list(result.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--backend", choices=["auto", "numpy", "tiled", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    results = [run_case(size, args.backend, args.seed) for size in args.sizes]
    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
