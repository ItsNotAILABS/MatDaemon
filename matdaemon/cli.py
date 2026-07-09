"""Command line interface for MatDaemon."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .matdaemon import matmul


def _cmd_matmul(args: argparse.Namespace) -> int:
    A = np.load(args.matrix_a)
    B = np.load(args.matrix_b)
    start = time.perf_counter()
    result = matmul(A, B, backend=args.backend)
    duration = time.perf_counter() - start
    np.save(args.output, result)
    print(json.dumps({"status": "ok", "backend": args.backend, "output": str(Path(args.output)), "shape": list(result.shape), "duration_seconds": round(duration, 6)}, indent=2))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    rows = args.size
    inner = args.inner or args.size
    cols = args.cols or args.size
    A = rng.standard_normal((rows, inner), dtype=np.float32)
    B = rng.standard_normal((inner, cols), dtype=np.float32)
    start = time.perf_counter()
    result = matmul(A, B, backend=args.backend)
    duration = time.perf_counter() - start
    print(json.dumps({"status": "ok", "backend": args.backend, "a_shape": list(A.shape), "b_shape": list(B.shape), "output_shape": list(result.shape), "duration_seconds": round(duration, 6)}, indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install API support with `pip install matdaemon[api]`.") from exc
    uvicorn.run("matdaemon.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from .mcp_server import main as run_mcp
    except Exception as exc:
        raise RuntimeError("Install MCP support with `pip install matdaemon[mcp]`.") from exc
    run_mcp()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MatDaemon AI matrix compute platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    matmul_parser = subparsers.add_parser("matmul", help="Multiply two .npy matrix files")
    matmul_parser.add_argument("matrix_a")
    matmul_parser.add_argument("matrix_b")
    matmul_parser.add_argument("--output", "-o", default="matdaemon_result.npy")
    matmul_parser.add_argument("--backend", choices=["auto", "numpy", "tiled", "cuda"], default="auto")
    matmul_parser.set_defaults(func=_cmd_matmul)

    bench_parser = subparsers.add_parser("benchmark", help="Run a quick synthetic benchmark")
    bench_parser.add_argument("--size", type=int, default=512)
    bench_parser.add_argument("--inner", type=int)
    bench_parser.add_argument("--cols", type=int)
    bench_parser.add_argument("--seed", type=int, default=7)
    bench_parser.add_argument("--backend", choices=["auto", "numpy", "tiled", "cuda"], default="auto")
    bench_parser.set_defaults(func=_cmd_benchmark)

    serve_parser = subparsers.add_parser("serve", help="Run the MatDaemon HTTP API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=_cmd_serve)

    mcp_parser = subparsers.add_parser("mcp", help="Run the MatDaemon MCP server over stdio")
    mcp_parser.set_defaults(func=_cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
