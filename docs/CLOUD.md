# MatDaemon Cloud and Coding Platform Guide

MatDaemon now exposes the same bounded tool suite through two product surfaces:

- local MCP stdio: `matdaemon mcp`
- hosted HTTP tool API: `GET /v1/tools` and `POST /v1/tools/{tool_name}`

This lets different coding and cloud environments use MatDaemon without giving them shell access.

## Platform Modes

| Environment | Best Surface | Why |
| --- | --- | --- |
| Local coding assistant | MCP stdio | The client can spawn `matdaemon mcp` locally. |
| Cursor / Claude Desktop / VS Code-style MCP client | MCP stdio | Uses standard MCP server config. |
| Codex-like local coding agent | MCP stdio or Python SDK | Can call the local server or import the package. |
| Hosted coding platform | HTTP tool API | Many hosted systems cannot spawn arbitrary local stdio processes. |
| Serverless workflow | HTTP tool API | Tool calls can be made as normal HTTP requests. |
| GitHub Actions | GitHub Action or HTTP API | Benchmarks can run directly in Actions. |
| Product backend | SDK or HTTP API | Use SDK in-process or deploy the API beside the app. |

## HTTP Tool API

Run the API:

```bash
matdaemon serve --host 0.0.0.0 --port 8000
```

List tools:

```bash
curl http://localhost:8000/v1/tools
```

Call matrix multiplication:

```bash
curl -X POST http://localhost:8000/v1/tools/matdaemon_matmul \
  -H 'content-type: application/json' \
  -d '{"arguments": {"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]], "backend": "numpy"}}'
```

Generate a GitHub Actions benchmark snippet:

```bash
curl -X POST http://localhost:8000/v1/tools/matdaemon_generate_github_action \
  -H 'content-type: application/json' \
  -d '{"arguments": {"profile": "ai", "backends": "numpy tiled", "repetitions": 1}}'
```

Run a bounded smoke benchmark:

```bash
curl -X POST http://localhost:8000/v1/tools/matdaemon_smoke_benchmark \
  -H 'content-type: application/json' \
  -d '{"arguments": {"size": 128, "backend": "auto"}}'
```

## MCP Stdio Config

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

Windows editable clone fallback:

```json
{
  "mcpServers": {
    "matdaemon": {
      "command": "python",
      "args": ["-m", "matdaemon.mcp_server"],
      "cwd": "C:\\Users\\Medin\\MatDaemon"
    }
  }
}
```

## What Cloud Platforms Can Do

### Discover capabilities

Call `matdaemon_platform_manifest` to understand package version, surfaces, use cases, proof gates, install commands, and tool API routes.

### Inspect runtime

Call `matdaemon_backend_status` to detect Python, platform, NumPy, and backend availability before choosing `auto`, `numpy`, `tiled`, or `cuda`.

### Validate payloads

Call `matdaemon_validate_matrices` before execution to catch dimension mismatches and return output shape.

### Run bounded compute

Call `matdaemon_matmul` for direct matrix multiplication and `matdaemon_similarity_top_k` for embedding ranking.

### Rank raw text without a separate embedding model

Call `matdaemon_embed_text` to turn strings into deterministic float vectors (signed hashing trick — no model download, no network, reproducible across processes), or `matdaemon_text_similarity_top_k` to embed and rank candidate strings against queries in one call. The vectors are lexical (shared words / optional character n-grams), so this covers dedup, fuzzy record linking, and retrieval over short text; swap in neural embeddings via `matdaemon_similarity_top_k` when you need semantic matching — the ranking math is identical.

```bash
curl -X POST http://localhost:8000/v1/tools/matdaemon_text_similarity_top_k \
  -H 'content-type: application/json' \
  -d '{"arguments": {"queries": ["quarterly financials report"], "candidates": ["Q3 financial report", "website redesign"], "k": 1}}'
```

### Train a model on the matmul backend

Call `matdaemon_train_classifier` to train a real classifier by gradient descent — logistic regression or a one-hidden-layer ReLU MLP — with every forward (`X @ W`) and backward (`X.T @ error`, `H.T @ delta`) pass running through matmul. It returns the learned weights, the loss curve, and training accuracy. Training is seeded and reproducible; sizes are capped so it stays a bounded compute surface. Pure NumPy — no autograd framework, no downloaded weights.

```bash
curl -X POST http://localhost:8000/v1/tools/matdaemon_train_classifier \
  -H 'content-type: application/json' \
  -d '{"arguments": {"features": [[0,0],[0,1],[1,0],[1,1]], "labels": [0,1,1,0], "model": "mlp", "epochs": 2000, "hidden": 8}}'
```

### Generate integration artifacts

Call `matdaemon_generate_api_payload` to create API request bodies and `matdaemon_generate_github_action` to create benchmark workflow snippets.

### Produce proof

Call `matdaemon_smoke_benchmark` for a capped local proof run, or use the GitHub Action benchmark surface for release artifacts.

## Safety Boundary

MatDaemon tools do not execute shell commands, read arbitrary files, mutate repositories, or make network calls. Cloud and coding platforms receive a compute/proof surface, not host control.

## Useful Suites to Build On Top

- **Agent Memory Suite:** `similarity_top_k` for memory routing, use-case metadata, and benchmark proof.
- **RAG Similarity Suite:** local embedding ranking without vector database dependency.
- **CI Benchmark Suite:** generated GitHub Action plus Markdown/JSON artifacts.
- **Cloud Tool Gateway:** `/v1/tools` for hosted agents that cannot use stdio MCP.
- **GPU Proof Suite:** backend inspection plus CUDA benchmark profile on GPU runners.
- **Model Training Suite:** `train_classifier` runs real gradient-descent training (logistic regression, MLP) with forward and backward passes on the matmul backend — the same engine can serve as an in-boundary trainer, not just an inference primitive.
