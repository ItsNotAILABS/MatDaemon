"""Comprehensive MatDaemon benchmark suite.

Runs AI-shaped matrix cases, compares backends, validates against NumPy, and
emits JSON plus Markdown reports for GitHub Actions artifacts, releases, and
benchmark docs.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import matdaemon as md


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    m: int
    k: int
    n: int
    dtype: str = "float32"

    @property
    def flops(self) -> int:
        return 2 * self.m * self.k * self.n

    @property
    def output_bytes(self) -> int:
        return self.m * self.n * np.dtype(self.dtype).itemsize


@dataclass
class BenchmarkResult:
    case: dict
    backend: str
    repetitions: int
    duration_seconds_min: float
    duration_seconds_median: float
    duration_seconds_mean: float
    gflops_median: float
    output_mb: float
    max_abs_error_vs_numpy: float | None
    status: str
    error: str | None = None


def default_cases(profile: str) -> list[BenchmarkCase]:
    if profile == "quick":
        return [
            BenchmarkCase("tiny-smoke", 64, 64, 64),
            BenchmarkCase("small-square", 256, 256, 256),
            BenchmarkCase("embedding-batch", 128, 768, 768),
        ]
    if profile == "ai":
        return [
            BenchmarkCase("embedding-projection", 256, 768, 1536),
            BenchmarkCase("attention-block", 512, 1024, 512),
            BenchmarkCase("rag-similarity", 1024, 768, 2048),
            BenchmarkCase("agent-memory-scan", 2048, 384, 2048),
        ]
    return [
        BenchmarkCase("small-square", 256, 256, 256),
        BenchmarkCase("medium-square", 512, 512, 512),
        BenchmarkCase("large-square", 1024, 1024, 1024),
        BenchmarkCase("wide-projection", 512, 2048, 1024),
        BenchmarkCase("embedding-batch", 2048, 768, 1024),
    ]


def parse_shape(shape: str) -> BenchmarkCase:
    label, dims = shape.split(":", 1) if ":" in shape else ("custom", shape)
    parts = [int(part) for part in dims.lower().replace("x", ",").split(",")]
    if len(parts) != 3:
        raise ValueError("Custom shape must be MxKxN, for example 1024x768x2048")
    return BenchmarkCase(label, parts[0], parts[1], parts[2])


def make_matrices(case: BenchmarkCase, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dtype = np.dtype(case.dtype)
    return (
        rng.standard_normal((case.m, case.k)).astype(dtype),
        rng.standard_normal((case.k, case.n)).astype(dtype),
    )


def time_backend(A: np.ndarray, B: np.ndarray, backend: str, repetitions: int) -> tuple[list[float], np.ndarray]:
    durations: list[float] = []
    output = None
    for _ in range(repetitions):
        start = time.perf_counter()
        output = md.matmul(A, B, backend=backend)
        durations.append(time.perf_counter() - start)
    assert output is not None
    return durations, output


def run_case(case: BenchmarkCase, backend: str, repetitions: int, seed: int, verify: bool) -> BenchmarkResult:
    A, B = make_matrices(case, seed)
    try:
        durations, output = time_backend(A, B, backend, repetitions)
        max_abs_error = None
        if verify:
            baseline = np.matmul(A, B)
            max_abs_error = float(np.max(np.abs(output - baseline)))
        median = statistics.median(durations)
        gflops = case.flops / median / 1e9 if median > 0 else math.inf
        return BenchmarkResult(
            case=asdict(case),
            backend=backend,
            repetitions=repetitions,
            duration_seconds_min=round(min(durations), 6),
            duration_seconds_median=round(median, 6),
            duration_seconds_mean=round(statistics.mean(durations), 6),
            gflops_median=round(gflops, 3),
            output_mb=round(case.output_bytes / (1024**2), 3),
            max_abs_error_vs_numpy=max_abs_error,
            status="ok",
        )
    except Exception as exc:
        return BenchmarkResult(
            case=asdict(case),
            backend=backend,
            repetitions=repetitions,
            duration_seconds_min=0.0,
            duration_seconds_median=0.0,
            duration_seconds_mean=0.0,
            gflops_median=0.0,
            output_mb=round(case.output_bytes / (1024**2), 3),
            max_abs_error_vs_numpy=None,
            status="error",
            error=str(exc),
        )


def markdown_report(payload: dict) -> str:
    lines = [
        "# MatDaemon Benchmark Report",
        "",
        "## Environment",
        "",
        f"- Python: `{payload['environment']['python']}`",
        f"- Platform: `{payload['environment']['platform']}`",
        f"- NumPy: `{payload['environment']['numpy']}`",
        f"- CUDA available through MatDaemon: `{payload['environment']['cuda_available']}`",
        "",
        "## Results",
        "",
        "| Case | Shape MxKxN | Backend | Median s | GFLOP/s | Output MB | Error vs NumPy | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in payload["results"]:
        case = result["case"]
        shape = f"{case['m']}x{case['k']}x{case['n']}"
        err = result["max_abs_error_vs_numpy"]
        err_text = "" if err is None else f"{err:.6g}"
        lines.append(
            f"| {case['name']} | {shape} | {result['backend']} | "
            f"{result['duration_seconds_median']} | {result['gflops_median']} | "
            f"{result['output_mb']} | {err_text} | {result['status']} |"
        )
    lines.extend(["", "## Reproduce", "", "```bash", payload["command"], "```"])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark-results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "benchmark-results.md").write_text(markdown_report(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MatDaemon benchmark profiles")
    parser.add_argument("--profile", choices=["quick", "launch", "ai"], default="launch")
    parser.add_argument("--quick", action="store_true", help="Alias for --profile quick --repetitions 1")
    parser.add_argument("--backends", nargs="+", choices=["auto", "numpy", "tiled", "cuda"], default=["numpy", "tiled"])
    parser.add_argument("--shape", action="append", help="Custom shape label:MxKxN, repeatable")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-verify", action="store_true", help="Skip NumPy correctness comparison")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any benchmark case errors")
    parser.add_argument("--output", type=Path, help="Directory for JSON and Markdown reports")
    args = parser.parse_args(argv)

    if args.quick:
        args.profile = "quick"
        args.repetitions = 1

    cases = default_cases(args.profile)
    if args.shape:
        cases.extend(parse_shape(shape) for shape in args.shape)

    results = [
        asdict(run_case(case, backend, args.repetitions, args.seed, verify=not args.no_verify))
        for case in cases
        for backend in args.backends
    ]
    payload = {
        "command": "python " + " ".join(sys.argv),
        "profile": args.profile,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cuda_available": md.cuda_available(),
        },
        "results": results,
    }

    print(json.dumps(payload, indent=2))
    if args.output:
        write_outputs(payload, args.output)

    has_errors = any(result["status"] == "error" for result in results)
    return 1 if args.strict and has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
