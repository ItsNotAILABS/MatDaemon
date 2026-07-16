"""Self-contained MCP server for MatDaemon AI tooling.

Run with:
    matdaemon mcp

The server speaks JSON-RPC over MCP stdio framing and intentionally avoids
external MCP runtime dependencies so it remains easy to install on Windows ARM,
CI runners, and minimal coding-agent environments.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from typing import Any, Callable, Literal

import numpy as np

from .matdaemon import cuda_available, matmul, validate_matrices
from .physics import (
    PHYSICS_ALGORITHMS,
    boltzmann_distribution,
    get_physics_algorithm,
    ising_energy,
    pairwise_distances,
)
from .platform import get_platform_manifest
from .text import hashing_embed, text_similarity_top_k
from .use_cases import USE_CASES, get_use_case

Backend = Literal["auto", "numpy", "tiled", "cuda"]
DType = Literal["float32", "float64"]
ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "MatDaemon", "version": "0.3.1"}


def _json_text(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}


def _matrix(values: list[list[float]], dtype: str) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def tool_platform_manifest(arguments: dict[str, Any]) -> dict[str, Any]:
    return get_platform_manifest()


def tool_backend_status(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "backends": {
            "auto": {"available": True, "description": "Routes to CUDA when available, otherwise CPU."},
            "numpy": {"available": True, "description": "Direct NumPy matmul path."},
            "tiled": {"available": True, "description": "Memory-aware block CPU path."},
            "cuda": {"available": cuda_available(), "description": "Optional CuPy RawKernel backend."},
        },
    }


def tool_validate_matrices(arguments: dict[str, Any]) -> dict[str, Any]:
    dtype = arguments.get("dtype", "float32")
    A = _matrix(arguments.get("a", []), dtype)
    B = _matrix(arguments.get("b", []), dtype)
    try:
        validate_matrices(A, B)
    except Exception as exc:
        return {"valid": False, "error": str(exc), "a_shape": list(A.shape), "b_shape": list(B.shape)}
    return {
        "valid": True,
        "a_shape": list(A.shape),
        "b_shape": list(B.shape),
        "output_shape": [int(A.shape[0]), int(B.shape[1])],
        "dtype": str(A.dtype),
    }


def tool_matmul(arguments: dict[str, Any]) -> dict[str, Any]:
    dtype = arguments.get("dtype", "float32")
    backend = arguments.get("backend", "auto")
    A = _matrix(arguments.get("a", []), dtype)
    B = _matrix(arguments.get("b", []), dtype)
    start = time.perf_counter()
    result = matmul(A, B, backend=backend)
    duration = time.perf_counter() - start
    return {
        "backend": backend,
        "duration_seconds": round(duration, 6),
        "a_shape": list(A.shape),
        "b_shape": list(B.shape),
        "shape": list(result.shape),
        "result": result.tolist(),
    }


def tool_similarity_top_k(arguments: dict[str, Any]) -> dict[str, Any]:
    backend = arguments.get("backend", "auto")
    k = int(arguments.get("k", 5))
    Q = np.asarray(arguments.get("queries", []), dtype=np.float32)
    C = np.asarray(arguments.get("candidates", []), dtype=np.float32)
    if Q.ndim != 2 or C.ndim != 2:
        raise ValueError("queries and candidates must be 2D arrays")
    if Q.shape[1] != C.shape[1]:
        raise ValueError("queries and candidates must have the same embedding dimension")
    k = max(1, min(k, C.shape[0]))
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    scores = matmul(Q, C.T, backend=backend)
    top_k = np.argsort(-scores, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top_k, axis=1)
    return {
        "top_k": top_k.tolist(),
        "top_scores": top_scores.tolist(),
        "scores_shape": list(scores.shape),
        "backend": backend,
    }


def tool_embed_text(arguments: dict[str, Any]) -> dict[str, Any]:
    vectors = hashing_embed(
        arguments.get("texts", []),
        dim=int(arguments.get("dim", 256)),
        word_ngram=int(arguments.get("word_ngram", 1)),
        char_ngram=int(arguments.get("char_ngram", 0)),
        normalize=bool(arguments.get("normalize", True)),
    )
    return {
        "vectors": vectors.tolist(),
        "shape": list(vectors.shape),
        "dim": int(arguments.get("dim", 256)),
        "deterministic": True,
    }


def tool_text_similarity_top_k(arguments: dict[str, Any]) -> dict[str, Any]:
    return text_similarity_top_k(
        arguments.get("queries", []),
        arguments.get("candidates", []),
        k=int(arguments.get("k", 5)),
        dim=int(arguments.get("dim", 256)),
        word_ngram=int(arguments.get("word_ngram", 1)),
        char_ngram=int(arguments.get("char_ngram", 0)),
        backend=arguments.get("backend", "auto"),
    )


def tool_use_cases(arguments: dict[str, Any]) -> dict[str, Any]:
    use_case_id = arguments.get("id")
    if use_case_id:
        use_case = get_use_case(str(use_case_id))
        if use_case is None:
            raise ValueError(f"Unknown use case: {use_case_id}")
        return {"use_case": use_case}
    return {"use_cases": USE_CASES}


def tool_physics_algorithms(arguments: dict[str, Any]) -> dict[str, Any]:
    algorithm_id = arguments.get("id")
    if algorithm_id:
        algorithm = get_physics_algorithm(str(algorithm_id))
        if algorithm is None:
            raise ValueError(f"Unknown physics algorithm: {algorithm_id}")
        return {"algorithm": algorithm}
    return {"algorithms": PHYSICS_ALGORITHMS, "count": len(PHYSICS_ALGORITHMS)}


def tool_pairwise_distances(arguments: dict[str, Any]) -> dict[str, Any]:
    backend = arguments.get("backend", "auto")
    dist = pairwise_distances(arguments.get("points", []), backend=backend)
    return {"distances": dist.tolist(), "shape": list(dist.shape), "backend": backend}


def tool_ising_energy(arguments: dict[str, Any]) -> dict[str, Any]:
    energy = ising_energy(
        arguments.get("spins", []),
        arguments.get("adjacency", []),
        coupling=float(arguments.get("coupling", 1.0)),
        field=float(arguments.get("field", 0.0)),
    )
    return {"energy": energy}


def tool_boltzmann_distribution(arguments: dict[str, Any]) -> dict[str, Any]:
    probs = boltzmann_distribution(arguments.get("energies", []), kt=float(arguments.get("kt", 1.0)))
    return {"probabilities": probs.tolist()}


def tool_generate_api_payload(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": "/v1/jobs/matmul" if arguments.get("async_job", True) else "/v1/matmul",
        "method": "POST",
        "headers": {"content-type": "application/json"},
        "body": {
            "a": arguments.get("a", [[1, 2], [3, 4]]),
            "b": arguments.get("b", [[5, 6], [7, 8]]),
            "backend": arguments.get("backend", "auto"),
            "dtype": arguments.get("dtype", "float32"),
            "use_case": arguments.get("use_case", "agent-memory-routing"),
        },
    }


def tool_generate_github_action(arguments: dict[str, Any]) -> dict[str, Any]:
    profile = arguments.get("profile", "ai")
    backends = arguments.get("backends", "numpy tiled")
    repetitions = str(arguments.get("repetitions", "1"))
    strict = str(arguments.get("strict", "true")).lower()
    yaml = (
        "- uses: ItsNotAILABS/MatDaemon/.github/actions/matdaemon-benchmark@main\n"
        "  with:\n"
        f"    profile: {profile}\n"
        f"    backends: {backends}\n"
        f"    repetitions: \"{repetitions}\"\n"
        f"    strict: \"{strict}\"\n"
    )
    return {"yaml": yaml}


def tool_smoke_benchmark(arguments: dict[str, Any]) -> dict[str, Any]:
    size = int(arguments.get("size", 128))
    backend = arguments.get("backend", "auto")
    seed = int(arguments.get("seed", 7))
    size = max(1, min(size, 1024))
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((size, size), dtype=np.float32)
    B = rng.standard_normal((size, size), dtype=np.float32)
    start = time.perf_counter()
    result = matmul(A, B, backend=backend)
    duration = time.perf_counter() - start
    gflops = (2 * size * size * size) / duration / 1e9 if duration > 0 else float("inf")
    return {
        "status": "ok",
        "backend": backend,
        "shape": [size, size, size],
        "duration_seconds": round(duration, 6),
        "gflops": round(gflops, 3),
        "output_shape": list(result.shape),
    }


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "matdaemon_platform_manifest": tool_platform_manifest,
    "matdaemon_backend_status": tool_backend_status,
    "matdaemon_validate_matrices": tool_validate_matrices,
    "matdaemon_matmul": tool_matmul,
    "matdaemon_similarity_top_k": tool_similarity_top_k,
    "matdaemon_embed_text": tool_embed_text,
    "matdaemon_text_similarity_top_k": tool_text_similarity_top_k,
    "matdaemon_use_cases": tool_use_cases,
    "matdaemon_physics_algorithms": tool_physics_algorithms,
    "matdaemon_pairwise_distances": tool_pairwise_distances,
    "matdaemon_ising_energy": tool_ising_energy,
    "matdaemon_boltzmann_distribution": tool_boltzmann_distribution,
    "matdaemon_generate_api_payload": tool_generate_api_payload,
    "matdaemon_generate_github_action": tool_generate_github_action,
    "matdaemon_smoke_benchmark": tool_smoke_benchmark,
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "matdaemon_platform_manifest",
        "description": "Return MatDaemon product surfaces, runtime stack, install commands, and proof gates.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "matdaemon_backend_status",
        "description": "Inspect available MatDaemon backends and runtime environment.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "matdaemon_validate_matrices",
        "description": "Validate two matrix payloads and return output shape or a validation error.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "b": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "dtype": {"type": "string", "enum": ["float32", "float64"], "default": "float32"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
    {
        "name": "matdaemon_matmul",
        "description": "Multiply two matrices and return result plus timing metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "b": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "backend": {"type": "string", "enum": ["auto", "numpy", "tiled", "cuda"], "default": "auto"},
                "dtype": {"type": "string", "enum": ["float32", "float64"], "default": "float32"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
    {
        "name": "matdaemon_similarity_top_k",
        "description": "Rank candidate embeddings for query embeddings and return top-k indexes and scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "candidates": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "k": {"type": "integer", "minimum": 1, "default": 5},
                "backend": {"type": "string", "enum": ["auto", "numpy", "tiled", "cuda"], "default": "auto"},
            },
            "required": ["queries", "candidates"],
            "additionalProperties": False,
        },
    },
    {
        "name": "matdaemon_embed_text",
        "description": "Embed text into deterministic float vectors via the signed hashing trick (no model download, no network, reproducible across processes).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "texts": {"type": "array", "items": {"type": "string"}},
                "dim": {"type": "integer", "minimum": 1, "default": 256},
                "word_ngram": {"type": "integer", "minimum": 1, "default": 1},
                "char_ngram": {"type": "integer", "minimum": 0, "default": 0},
                "normalize": {"type": "boolean", "default": True},
            },
            "required": ["texts"],
            "additionalProperties": False,
        },
    },
    {
        "name": "matdaemon_text_similarity_top_k",
        "description": "Rank candidate strings against query strings by lexical cosine similarity (hashing-embed + top-k in one call).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}},
                "candidates": {"type": "array", "items": {"type": "string"}},
                "k": {"type": "integer", "minimum": 1, "default": 5},
                "dim": {"type": "integer", "minimum": 1, "default": 256},
                "word_ngram": {"type": "integer", "minimum": 1, "default": 1},
                "char_ngram": {"type": "integer", "minimum": 0, "default": 0},
                "backend": {"type": "string", "enum": ["auto", "numpy", "tiled", "cuda"], "default": "auto"},
            },
            "required": ["queries", "candidates"],
            "additionalProperties": False,
        },
    },
    {
        "name": "matdaemon_use_cases",
        "description": "List all AI use cases, or fetch one use case by id.",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "additionalProperties": False},
    },
    {
        "name": "matdaemon_physics_algorithms",
        "description": "List the physics algorithm registry (formula, matrix form, and matmul role for each), or fetch one by id.",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "additionalProperties": False},
    },
    {
        "name": "matdaemon_pairwise_distances",
        "description": "Euclidean distance matrix of a point cloud via the Gram-matrix matmul identity (many-body / N-body primitive).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "backend": {"type": "string", "enum": ["auto", "numpy", "tiled", "cuda"], "default": "auto"},
            },
            "required": ["points"],
            "additionalProperties": False,
        },
    },
    {
        "name": "matdaemon_ising_energy",
        "description": "Ising Hamiltonian energy E = -J/2 s^T A s - h sum(s) for a spin configuration on an adjacency matrix.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "spins": {"type": "array", "items": {"type": "number"}},
                "adjacency": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "coupling": {"type": "number", "default": 1.0},
                "field": {"type": "number", "default": 0.0},
            },
            "required": ["spins", "adjacency"],
            "additionalProperties": False,
        },
    },
    {
        "name": "matdaemon_boltzmann_distribution",
        "description": "Boltzmann occupation probabilities p_i = exp(-E_i/kT)/Z for a set of energy levels.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "energies": {"type": "array", "items": {"type": "number"}},
                "kt": {"type": "number", "default": 1.0},
            },
            "required": ["energies"],
            "additionalProperties": False,
        },
    },
    {
        "name": "matdaemon_generate_api_payload",
        "description": "Generate a ready-to-send HTTP API payload for MatDaemon matrix jobs.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": True},
    },
    {
        "name": "matdaemon_generate_github_action",
        "description": "Generate a GitHub Actions snippet for the MatDaemon benchmark action.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": True},
    },
    {
        "name": "matdaemon_smoke_benchmark",
        "description": "Run a bounded local square-matrix smoke benchmark. Size is capped at 1024.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "size": {"type": "integer", "minimum": 1, "maximum": 1024, "default": 128},
                "backend": {"type": "string", "enum": ["auto", "numpy", "tiled", "cuda"], "default": "auto"},
                "seed": {"type": "integer", "default": 7},
            },
            "additionalProperties": False,
        },
    },
]


def _response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")

    if method == "initialize":
        return _response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _response(request_id, {"tools": TOOLS})
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return _error(request_id, -32601, f"Unknown tool: {name}")
        try:
            return _response(request_id, _json_text(handler(arguments)))
        except Exception as exc:
            return _response(request_id, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
    return _error(request_id, -32601, f"Unsupported method: {method}")


def _read_message(stdin: Any) -> dict[str, Any] | None:
    header = stdin.buffer.readline()
    if not header:
        return None

    if header.startswith(b"Content-Length:"):
        length = int(header.decode("ascii").split(":", 1)[1].strip())
        while True:
            line = stdin.buffer.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        body = stdin.buffer.read(length)
        return json.loads(body.decode("utf-8"))

    stripped = header.strip()
    if not stripped:
        return None
    return json.loads(stripped.decode("utf-8"))


def _write_message(stdout: Any, message: dict[str, Any]) -> None:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stdout.buffer.write(body)
    stdout.buffer.flush()


def serve_stdio() -> None:
    while True:
        message = _read_message(sys.stdin)
        if message is None:
            break
        response = handle_request(message)
        if response is not None:
            _write_message(sys.stdout, response)


def create_mcp_server() -> dict[str, Any]:
    """Return the self-contained MCP server contract for tests and introspection."""
    return {"protocolVersion": PROTOCOL_VERSION, "serverInfo": SERVER_INFO, "tools": TOOLS}


def main() -> None:
    serve_stdio()


if __name__ == "__main__":
    main()
