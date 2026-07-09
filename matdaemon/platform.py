"""Product platform manifest for MatDaemon surfaces.

The manifest is intentionally plain data so the SDK, API, CLI, MCP server,
docs, and launch surfaces can describe the same platform contract.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

from .use_cases import USE_CASES


def _package_version() -> str:
    try:
        return metadata.version("matdaemon")
    except metadata.PackageNotFoundError:
        return "0.3.0"


MCP_TOOLS = [
    "matdaemon_platform_manifest",
    "matdaemon_backend_status",
    "matdaemon_validate_matrices",
    "matdaemon_matmul",
    "matdaemon_similarity_top_k",
    "matdaemon_use_cases",
    "matdaemon_generate_api_payload",
    "matdaemon_generate_github_action",
    "matdaemon_smoke_benchmark",
]

PLATFORM_SURFACES: list[dict[str, str]] = [
    {
        "id": "sdk",
        "title": "Python SDK",
        "entrypoint": "import matdaemon as md",
        "contract": "md.matmul(A, B, backend='auto')",
        "operator": "developer, agent runtime, notebook, worker",
        "value": "Embed matrix compute directly into AI and scientific Python code.",
    },
    {
        "id": "daemon",
        "title": "In-process daemon",
        "entrypoint": "MatDaemon()",
        "contract": "submit MatrixTask objects and collect MatrixResult outputs",
        "operator": "local worker, simulation loop, agent runtime",
        "value": "Queue matrix jobs without introducing an external service dependency.",
    },
    {
        "id": "cli",
        "title": "Command line interface",
        "entrypoint": "matdaemon",
        "contract": "matdaemon matmul | benchmark | serve | mcp | platform",
        "operator": "developer, CI runner, local automation",
        "value": "Run matrix jobs, smoke benchmarks, API service, MCP server, and manifest checks from a shell.",
    },
    {
        "id": "api",
        "title": "HTTP mini platform",
        "entrypoint": "matdaemon serve",
        "contract": "GET /health, GET /v1/platform, POST /v1/matmul, POST /v1/jobs/matmul",
        "operator": "service client, AI worker, remote job caller",
        "value": "Expose synchronous and async matrix jobs over a small FastAPI service.",
    },
    {
        "id": "mcp",
        "title": "Self-contained MCP server",
        "entrypoint": "matdaemon mcp",
        "contract": ", ".join(MCP_TOOLS),
        "operator": "MCP-compatible AI client or coding platform",
        "value": "Let coding agents discover, validate, benchmark, and invoke matrix compute over stdio without broad machine access.",
    },
    {
        "id": "github-action",
        "title": "GitHub Action",
        "entrypoint": ".github/actions/matdaemon-benchmark",
        "contract": "profile, backends, repetitions, strict inputs produce JSON and Markdown artifacts",
        "operator": "maintainer, release workflow, benchmark runner",
        "value": "Run AI-shaped benchmark proof directly from GitHub Actions.",
    },
    {
        "id": "cuda-backend",
        "title": "Optional CUDA backend",
        "entrypoint": "backend='cuda'",
        "contract": "CuPy RawKernel GEMM path with CPU install kept lightweight",
        "operator": "GPU host, benchmark runner, high-throughput worker",
        "value": "Use the specialized CUDA backend when a compatible GPU runtime exists.",
    },
]

RUNTIME_STACK: list[dict[str, str]] = [
    {"layer": "client", "role": "SDK, CLI, HTTP, MCP, or GitHub Action caller"},
    {"layer": "contracts", "role": "typed payloads, platform manifest, use-case registry, benchmark profiles"},
    {"layer": "orchestration", "role": "MatDaemon queue, API job lifecycle, CLI dispatch, MCP JSON-RPC tools"},
    {"layer": "compute", "role": "auto, numpy, tiled CPU, or optional CUDA RawKernel backend"},
    {"layer": "proof", "role": "unit tests, API lifecycle tests, MCP tool tests, benchmark JSON, benchmark Markdown, CI artifacts"},
]

PROOF_GATES: list[dict[str, str]] = [
    {"gate": "correctness", "evidence": "matrix outputs validated against NumPy-compatible expected results"},
    {"gate": "platform", "evidence": "health, platform manifest, use cases, sync jobs, and async jobs covered by API tests"},
    {"gate": "agent surface", "evidence": "self-contained MCP server exposes bounded JSON-RPC tools without external MCP dependencies"},
    {"gate": "benchmark", "evidence": "benchmark suite writes JSON and Markdown artifacts and supports strict failure mode"},
    {"gate": "packaging", "evidence": "pyproject metadata, optional extras, console script, Dockerfile, and GitHub Action surface"},
]


def get_platform_manifest() -> dict[str, Any]:
    """Return the public MatDaemon platform contract."""
    return {
        "name": "MatDaemon",
        "version": _package_version(),
        "status": "production-beta",
        "tagline": "AI-native matrix compute platform for agents, RAG, simulations, and ML automation.",
        "surfaces": PLATFORM_SURFACES,
        "runtime_stack": RUNTIME_STACK,
        "use_cases": USE_CASES,
        "mcp_tools": MCP_TOOLS,
        "proof_gates": PROOF_GATES,
        "install": {
            "source": "git clone https://github.com/ItsNotAILABS/MatDaemon.git && cd MatDaemon && python -m pip install -e .",
            "api": "python -m pip install -e .[api]",
            "mcp": "python -m pip install -e .[mcp]",
            "dev": "python -m pip install -e .[dev,api]",
            "pypi_status": "PyPI publishing is pending; use source install until the first package release is uploaded.",
        },
        "operator_commands": {
            "serve_api": "matdaemon serve --host 0.0.0.0 --port 8000",
            "run_mcp": "matdaemon mcp",
            "show_manifest": "matdaemon platform",
            "quick_benchmark": "python benchmarks/benchmark_suite.py --quick",
            "ai_benchmark": "python benchmarks/benchmark_suite.py --profile ai --backends auto numpy tiled --output benchmarks/results-ai",
        },
    }
