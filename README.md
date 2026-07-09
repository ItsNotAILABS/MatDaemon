<p align="center">
  <img src="docs/assets/matdaemon-platform.svg" alt="MatDaemon platform architecture" width="100%">
</p>

<h1 align="center">MatDaemon</h1>

<p align="center">
  AI-native matrix compute platform for agents, RAG systems, simulations, and ML automation.
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9+-2563eb.svg"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-f59e0b.svg"></a>
  <a href="#mcp-server"><img alt="MCP server" src="https://img.shields.io/badge/MCP-self--contained-111827.svg"></a>
  <a href="#http-tool-api"><img alt="HTTP tool API" src="https://img.shields.io/badge/HTTP-tool_API-0f766e.svg"></a>
  <a href="#github-callable-benchmarks"><img alt="GitHub callable" src="https://img.shields.io/badge/GitHub-callable-24292f.svg"></a>
  <a href="#cuda-backend"><img alt="CUDA optional" src="https://img.shields.io/badge/CUDA-optional-76B900.svg"></a>
</p>

MatDaemon packages matrix multiplication as a real product surface: Python SDK, async in-process daemon, CLI, HTTP job API, HTTP tool API, self-contained MCP server, GitHub Action, benchmark harness, Docker API surface, and optional CUDA RawKernel backend.

It is built for AI systems that need a small, callable compute layer without bringing in a full ML framework or inventing a one-off matrix service for every agent, RAG pipeline, simulation, or automation worker.

## Why It Exists

AI products keep needing the same matrix operations in different places: embedding similarity, memory routing, projection, attention-style score blocks, simulation transitions, and benchmarkable local compute. MatDaemon turns that repeated work into one installable platform with clear contracts and proof artifacts.

## Install

PyPI publishing is pending. Until the first package release is uploaded, install from GitHub or from a local clone:

```bash
python -m pip install "git+https://github.com/ItsNotAILABS/MatDaemon.git"
```

From source:

```bash
git clone https://github.com/ItsNotAILABS/MatDaemon.git
cd MatDaemon
python -m pip install -e .
python -m pip install -e .[dev,api]
pytest -q
```

After the first PyPI release is published:

```bash
python -m pip install matdaemon
python -m pip install "matdaemon[api]"
python -m pip install "matdaemon[mcp]"
```

Windows ARM note: the API extra uses plain `uvicorn`, not `uvicorn[standard]`, so it does not require compiling `httptools`. The MCP server is self-contained and does not require the external `mcp` package.

## Platform Surfaces

| Surface | Entry point | Production use |
| --- | --- | --- |
| SDK | `import matdaemon as md` | embed matrix compute in Python agents and pipelines |
| Daemon | `md.MatDaemon()` | queue in-process async matrix jobs |
| CLI | `matdaemon matmul`, `matdaemon benchmark`, `matdaemon platform` | local operator and CI workflows |
| HTTP API | `matdaemon serve` | sync and async matrix jobs over FastAPI |
| HTTP Tool API | `GET /v1/tools`, `POST /v1/tools/{tool_name}` | cloud platforms and hosted agents that cannot spawn stdio MCP |
| MCP Server | `matdaemon mcp` | tool-calling AI clients and coding platforms over stdio |
| GitHub Action | `.github/actions/matdaemon-benchmark` | benchmark MatDaemon from GitHub Actions |
| CUDA Backend | `backend="cuda"` | optional CuPy RawKernel GEMM on GPU hosts |

## First Run

```python
import numpy as np
import matdaemon as md

A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)
C = md.matmul(A, B, backend="auto")
```

Inspect the platform contract from the SDK or CLI:

```bash
matdaemon platform
```

```python
import matdaemon as md

manifest = md.get_platform_manifest()
print(manifest["surfaces"])
```

## HTTP Mini Platform

Run the API:

```bash
matdaemon serve --host 0.0.0.0 --port 8000
```

Discover the product contract:

```bash
curl http://localhost:8000/v1/platform
curl http://localhost:8000/v1/use-cases
```

Submit an async matrix job:

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

## HTTP Tool API

Cloud platforms, hosted agents, and serverless workflows can call the same bounded tool suite over HTTP:

```bash
curl http://localhost:8000/v1/tools
```

```bash
curl -X POST http://localhost:8000/v1/tools/matdaemon_similarity_top_k \
  -H 'content-type: application/json' \
  -d '{"arguments": {"queries": [[1, 0]], "candidates": [[1, 0], [0, 1]], "k": 1}}'
```

See [docs/CLOUD.md](docs/CLOUD.md).

Docker:

```bash
docker compose up --build
```

## MCP Server

MatDaemon includes a self-contained MCP stdio server for coding platforms and AI clients:

```bash
matdaemon mcp
```

MCP tools:

| Tool | Use |
| --- | --- |
| `matdaemon_platform_manifest` | return product surfaces, runtime stack, install commands, and proof gates |
| `matdaemon_backend_status` | inspect available backends and runtime environment |
| `matdaemon_validate_matrices` | validate matrix payloads before execution |
| `matdaemon_matmul` | multiply matrices and return result plus timing metadata |
| `matdaemon_similarity_top_k` | rank candidate embeddings for local RAG or memory routing |
| `matdaemon_use_cases` | list AI use cases and recommended backend shape |
| `matdaemon_generate_api_payload` | create a ready-to-send HTTP API job payload |
| `matdaemon_generate_github_action` | create a GitHub Action benchmark snippet |
| `matdaemon_smoke_benchmark` | run a bounded local benchmark for tool validation |

Generic MCP config:

```json
{
  "mcpServers": {
    "matdaemon": {
      "command": "matdaemon",
      "args": ["mcp"]
    }
  }
}
```

For a local editable clone on Windows, use the Python module path if the `matdaemon` console script is not on PATH:

```json
{
  "mcpServers": {
    "matdaemon": {
      "command": "python",
      "args": ["-m", "matdaemon.mcp_server"]
    }
  }
}
```

See [docs/MCP.md](docs/MCP.md).

## GitHub Callable Benchmarks

```yaml
- uses: ItsNotAILABS/MatDaemon/.github/actions/matdaemon-benchmark@main
  with:
    profile: ai
    backends: numpy tiled
    repetitions: "1"
    strict: "true"
```

The action produces JSON and Markdown benchmark artifacts that can be attached to releases, issues, launch posts, or hardware proof notes. See [docs/GITHUB_ACTION.md](docs/GITHUB_ACTION.md).

## AI Use Cases

| Use case | Shape | Recommended backend |
| --- | --- | --- |
| Agent memory routing | `queries[M, D] @ memories[N, D].T` | `auto` |
| Local RAG similarity | `queries[M, D] @ docs[N, D].T` | `auto` |
| Embedding projection | `embeddings[B, Din] @ weights[Din, Dout]` | `numpy` |
| Attention-style score blocks | `Q[T, D] @ K[S, D].T` | `tiled` |
| Simulation worker steps | `state[M, K] @ transition[K, N]` | `auto` |

Examples:

```bash
python examples/agent_embedding_router.py
python examples/local_rag_similarity.py
```

## Benchmarks

```bash
python benchmarks/benchmark_suite.py --quick
python benchmarks/benchmark_suite.py --profile ai --backends auto numpy tiled --output benchmarks/results-ai
python benchmarks/benchmark_suite.py --profile launch --backends numpy tiled --strict --output benchmarks/results
```

When `--output` is provided, MatDaemon writes:

- `benchmark-results.json`
- `benchmark-results.md`

## Publish

This repo includes `.github/workflows/publish.yml` for PyPI Trusted Publishing. After the PyPI pending publisher is configured for `ItsNotAILABS/MatDaemon` and workflow `publish.yml`, publish a GitHub release such as `v0.3.2` to upload the package.

See [docs/PUBLISHING.md](docs/PUBLISHING.md).

## CUDA Backend

MatDaemon preserves the specialized CUDA RawKernel backend under:

```text
backends/cuda_backend.py
```

The legacy misspelled path remains as a compatibility shim:

```text
backends/cude_backend.py
```

CPU installs stay lightweight. CUDA imports are optional and only required when `backend="cuda"` is requested on a compatible GPU host.

## Runtime Stack

```mermaid
flowchart LR
    clients[SDK / CLI / API / MCP / HTTP Tool API / GitHub Action]
    contracts[Platform manifest / use cases / tool schemas / benchmark profiles]
    orchestration[MatDaemon queue / API jobs / MCP tools / HTTP tools]
    compute[auto / numpy / tiled / cuda]
    proof[tests / CI / benchmark artifacts]

    clients --> contracts --> orchestration --> compute --> proof
```

## Production Proof Gates

| Gate | Evidence |
| --- | --- |
| Correctness | matrix outputs tested against expected NumPy-compatible results |
| Platform | health, manifest, use cases, sync jobs, async jobs, and HTTP tools covered by API tests |
| Agent surface | self-contained MCP server exposes bounded JSON-RPC tools without external MCP dependencies |
| Cloud surface | HTTP tool API exposes the MCP tool suite to hosted platforms without shell access |
| Benchmark | suite writes JSON and Markdown artifacts with strict failure mode |
| Packaging | package extras, console script, Dockerfile, publish workflow, docs, and GitHub Action surface |

## Docs

- [Platform guide](docs/PLATFORM.md)
- [MCP guide](docs/MCP.md)
- [Cloud and coding platform guide](docs/CLOUD.md)
- [Publishing guide](docs/PUBLISHING.md)
- [GitHub Action guide](docs/GITHUB_ACTION.md)
- [Benchmark guide](docs/BENCHMARKING.md)
- [Product surface](docs/PRODUCT.md)
- [Launch checklist](docs/LAUNCH.md)
- [Repository metadata](docs/METADATA.md)

## License

MIT License.
