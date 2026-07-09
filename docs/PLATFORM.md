# MatDaemon Platform

MatDaemon ships as a mini platform with six callable surfaces:

1. **SDK** - `import matdaemon as md` for direct matrix multiplication.
2. **Daemon** - `MatDaemon` for async in-process agent and worker jobs.
3. **CLI** - `matdaemon matmul`, `matdaemon benchmark`, `matdaemon serve`, and `matdaemon mcp`.
4. **HTTP API** - FastAPI service for synchronous and async matrix jobs.
5. **MCP Server** - tool surface for MCP-compatible AI clients.
6. **GitHub Action** - benchmark runs directly from GitHub Actions.

## Run the API

```bash
python -m pip install -e .[api]
matdaemon serve --host 0.0.0.0 --port 8000
```

Synchronous call:

```bash
curl -X POST http://localhost:8000/v1/matmul \
  -H 'content-type: application/json' \
  -d '{"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]], "backend": "auto"}'
```

Async job call:

```bash
curl -X POST http://localhost:8000/v1/jobs/matmul \
  -H 'content-type: application/json' \
  -d '{"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]], "backend": "numpy", "use_case": "agent-memory-routing"}'
```

Then poll:

```bash
curl http://localhost:8000/v1/jobs/<job_id>
curl http://localhost:8000/v1/jobs/<job_id>/result
```

Discover AI use cases:

```bash
curl http://localhost:8000/v1/use-cases
```

## Run the MCP Server

```bash
python -m pip install -e .[mcp]
matdaemon mcp
```

See [MCP.md](MCP.md).

## Docker

```bash
docker compose up --build
```

## GitHub Actions

Use **Actions -> matdaemon-benchmark -> Run workflow** to execute benchmark profiles from GitHub and download JSON/Markdown artifacts.

See [GITHUB_ACTION.md](GITHUB_ACTION.md).

## AI Platform Fit

MatDaemon is useful anywhere an AI system needs fast matrix operations without coupling itself to a full ML framework:

- agent memory routing
- local RAG similarity
- embedding projection
- attention-style blocks
- simulation workers
- benchmarkable local compute nodes

## Production Next Gates

- persistent external job queue
- artifact result storage
- streaming progress
- cancellation
- GPU runner profile
- hosted demo endpoint
