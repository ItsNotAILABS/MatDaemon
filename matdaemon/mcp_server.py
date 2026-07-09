"""MCP server for AI clients that need MatDaemon tools.

Install with:
    pip install matdaemon[mcp]

Run with:
    matdaemon mcp
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from .matdaemon import matmul
from .use_cases import USE_CASES

try:  # pragma: no cover - optional dependency import
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover
    FastMCP = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None

Backend = Literal["auto", "numpy", "tiled", "cuda"]
DType = Literal["float32", "float64"]


def create_mcp_server():
    if FastMCP is None:  # pragma: no cover
        raise RuntimeError("MCP support requires `pip install matdaemon[mcp]`.") from _IMPORT_ERROR

    mcp = FastMCP("MatDaemon")

    @mcp.tool()
    def matdaemon_matmul(a: list[list[float]], b: list[list[float]], backend: Backend = "auto", dtype: DType = "float32") -> dict:
        """Multiply two matrices for an AI workflow and return result plus timing metadata."""
        A = np.asarray(a, dtype=dtype)
        B = np.asarray(b, dtype=dtype)
        start = time.perf_counter()
        result = matmul(A, B, backend=backend)
        duration = time.perf_counter() - start
        return {"backend": backend, "duration_seconds": round(duration, 6), "shape": list(result.shape), "result": result.tolist()}

    @mcp.tool()
    def matdaemon_use_cases() -> dict:
        """List AI use cases where MatDaemon can be called by an agent."""
        return {"use_cases": USE_CASES}

    @mcp.tool()
    def matdaemon_similarity_top_k(queries: list[list[float]], candidates: list[list[float]], k: int = 5, backend: Backend = "auto") -> dict:
        """Return top-k candidate indexes for query/candidate embedding similarity."""
        Q = np.asarray(queries, dtype=np.float32)
        C = np.asarray(candidates, dtype=np.float32)
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
        scores = matmul(Q, C.T, backend=backend)
        top_k = np.argsort(-scores, axis=1)[:, :k]
        return {"top_k": top_k.tolist(), "scores_shape": list(scores.shape), "backend": backend}

    return mcp


def main() -> None:
    create_mcp_server().run()


if __name__ == "__main__":
    main()
