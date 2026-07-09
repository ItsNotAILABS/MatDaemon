# MatDaemon Platform

MatDaemon now ships as more than a Python helper. It has four product surfaces:

1. **SDK** - `import matdaemon as md` for direct matrix multiplication.
2. **Daemon** - `MatDaemon` for async agent and worker jobs.
3. **CLI** - `matdaemon matmul`, `matdaemon benchmark`, and `matdaemon serve`.
4. **API** - FastAPI service for HTTP matrix compute.

## Run the API

```bash
python -m pip install -e .[api]
matdaemon serve --host 0.0.0.0 --port 8000
```

Then call it:

```bash
curl -X POST http://localhost:8000/v1/matmul \
  -H 'content-type: application/json' \
  -d '{"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]], "backend": "auto"}'
```

## Docker

```bash
docker compose up --build
```

## AI Platform Fit

MatDaemon is useful anywhere an AI system needs fast matrix operations without coupling itself to a full ML framework:

- agent memory routing
- local RAG similarity
- embedding projection
- attention-style blocks
- simulation workers
- benchmarkable local compute nodes

## Production Next Gates

- persistent job queue
- artifact result storage
- streaming progress
- cancellation
- GPU runner profile
- hosted demo endpoint
