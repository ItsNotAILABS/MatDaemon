# MatDaemon Platform Guide

MatDaemon is a mini compute platform packaged as a Python project. The product is not only `matmul`; it is the collection of callable surfaces, contracts, runtime routes, proof gates, and operator workflows around matrix compute for AI systems.

![MatDaemon platform architecture](assets/matdaemon-platform.svg)

## Platform Contract

The platform contract is exposed from one shared manifest:

```bash
matdaemon platform
curl http://localhost:8000/v1/platform
```

```python
import matdaemon as md

manifest = md.get_platform_manifest()
```

The manifest includes:

- product name, version, status, and tagline
- SDK, daemon, CLI, API, MCP, HTTP tool API, GitHub Action, and CUDA surfaces
- runtime stack layers
- AI use-case registry
- MCP/HTTP tool registry
- proof gates
- source and PyPI install posture
- operator commands

## Install Posture

PyPI publishing is pending. Use source install until the first package release is uploaded:

```bash
git clone https://github.com/ItsNotAILABS/MatDaemon.git
cd MatDaemon
python -m pip install -e .
python -m pip install -e .[dev,api]
pytest -q
```

Windows ARM safe path:

```bash
python -m pip install -e .[dev,api]
pytest -q
matdaemon mcp
```

The API path uses plain `uvicorn`, not `uvicorn[standard]`, and the MCP server is self-contained.

## Runtime Architecture

```mermaid
flowchart TD
    A[AI client, developer, CI, cloud platform, or service] --> B[SDK, CLI, HTTP API, HTTP Tool API, MCP, or GitHub Action]
    B --> C[platform manifest and typed contracts]
    C --> D[MatDaemon orchestration]
    D --> E{backend selection}
    E --> F[NumPy]
    E --> G[tiled CPU]
    E --> H[CUDA RawKernel]
    D --> I[status, results, benchmark artifacts]
```

## Operator Surfaces

| Operator | Surface | Command or contract |
| --- | --- | --- |
| Developer | SDK | `md.matmul(A, B, backend="auto")` |
| Agent runtime | MCP | `matdaemon_matmul`, `matdaemon_similarity_top_k`, `matdaemon_backend_status` |
| Coding platform | MCP | `matdaemon_platform_manifest`, `matdaemon_generate_api_payload`, `matdaemon_generate_github_action` |
| Cloud platform | HTTP Tool API | `GET /v1/tools`, `POST /v1/tools/{tool_name}` |
| Service caller | HTTP API | `POST /v1/jobs/matmul`, `GET /v1/jobs/{job_id}` |
| Maintainer | GitHub Action | `.github/actions/matdaemon-benchmark` |
| Release operator | PyPI workflow | `.github/workflows/publish.yml` |
| GPU operator | CUDA backend | `backend="cuda"` |
| Benchmark operator | Benchmark suite | `benchmark-results.json`, `benchmark-results.md` |

## HTTP API

Run the service:

```bash
python -m pip install -e .[api]
matdaemon serve --host 0.0.0.0 --port 8000
```

Core endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | service health and job counters |
| `/v1/platform` | `GET` | product manifest and runtime contract |
| `/v1/tools` | `GET` | HTTP tool schema discovery |
| `/v1/tools/{tool_name}` | `POST` | call bounded MatDaemon tools over HTTP |
| `/v1/use-cases` | `GET` | AI use-case registry |
| `/v1/matmul` | `POST` | synchronous matrix multiplication |
| `/v1/jobs/matmul` | `POST` | create async matrix job |
| `/v1/jobs/{job_id}` | `GET` | poll job status |
| `/v1/jobs/{job_id}/result` | `GET` | fetch completed result |

Async job flow:

```bash
curl -X POST http://localhost:8000/v1/jobs/matmul \
  -H 'content-type: application/json' \
  -d '{"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]], "backend": "numpy", "use_case": "agent-memory-routing"}'

curl http://localhost:8000/v1/jobs/<job_id>
curl http://localhost:8000/v1/jobs/<job_id>/result
```

HTTP tool call:

```bash
curl -X POST http://localhost:8000/v1/tools/matdaemon_matmul \
  -H 'content-type: application/json' \
  -d '{"arguments": {"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]], "backend": "numpy"}}'
```

## MCP Server

Run the self-contained stdio server:

```bash
python -m pip install -e .[mcp]
matdaemon mcp
```

MCP and HTTP tools:

- `matdaemon_platform_manifest`
- `matdaemon_backend_status`
- `matdaemon_validate_matrices`
- `matdaemon_matmul`
- `matdaemon_similarity_top_k`
- `matdaemon_use_cases`
- `matdaemon_generate_api_payload`
- `matdaemon_generate_github_action`
- `matdaemon_smoke_benchmark`

Security posture: the MCP server and HTTP tool API expose bounded matrix compute, benchmark, and discovery helpers. They do not execute shell commands, read arbitrary files, mutate repositories, or access network resources.

## Docker

```bash
docker compose up --build
curl http://localhost:8000/v1/platform
curl http://localhost:8000/v1/tools
```

## GitHub Actions

Use **Actions -> matdaemon-benchmark -> Run workflow** to execute benchmark profiles from GitHub and download JSON/Markdown artifacts.

Use **GitHub Releases -> Publish release** after configuring PyPI Trusted Publishing to publish package artifacts through `.github/workflows/publish.yml`.

See [GITHUB_ACTION.md](GITHUB_ACTION.md) and [PUBLISHING.md](PUBLISHING.md).

## Production Posture

Current production-ready surfaces:

- source-installable Python package metadata and optional extras
- PyPI trusted publishing workflow
- SDK and daemon APIs
- CLI for local operator workflows
- FastAPI mini platform with sync and async jobs
- HTTP tool API for hosted/cloud platforms
- self-contained MCP server for AI clients and coding platforms
- GitHub Action benchmark runner
- Docker API surface
- benchmark suite with strict mode and artifact output
- CUDA backend preserved as optional GPU path
- CI-covered platform manifest, MCP tools, HTTP tools, and API lifecycle

## Next Production Gates

These are future hardening gates, not blockers for shipping the current source package:

- PyPI release upload
- hosted demo endpoint
- authentication/rate limiting for public HTTP deployments
- persistent external job queue for multi-process deployments
- result artifact storage for large matrix outputs
- streaming progress and cancellation
- signed release artifacts
- GPU runner benchmark table
