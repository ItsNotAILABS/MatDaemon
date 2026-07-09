# MatDaemon MCP Server

MatDaemon exposes bounded matrix compute, platform discovery, benchmark, and payload-generation tools to MCP-compatible AI clients and coding platforms.

The server is self-contained. It speaks JSON-RPC over MCP stdio framing and does not require the external `mcp` Python package. That keeps installs lighter on Windows ARM, CI runners, and minimal coding-agent environments.

## Install

PyPI publishing is pending, so install from source for now:

```bash
git clone https://github.com/ItsNotAILABS/MatDaemon.git
cd MatDaemon
python -m pip install -e .[mcp]
```

For development and tests:

```bash
python -m pip install -e .[dev,api]
pytest -q
```

## Run

```bash
matdaemon mcp
```

If the console script is not on PATH:

```bash
python -m matdaemon.mcp_server
```

## Client Configs

Generic MCP shape:

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

Editable clone / Windows fallback:

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

Project-local Python example:

```json
{
  "mcpServers": {
    "matdaemon": {
      "command": "C:\\Users\\Medin\\AppData\\Local\\Programs\\Python\\Python311-arm64\\python.exe",
      "args": ["-m", "matdaemon.mcp_server"],
      "cwd": "C:\\Users\\Medin\\MatDaemon"
    }
  }
}
```

Use the generic shape for Cursor, Claude Desktop, VS Code-compatible MCP clients, Codex-like local agents, and other coding tools that accept stdio MCP server configs.

## Tools

| Tool | Purpose |
| --- | --- |
| `matdaemon_platform_manifest` | Return product surfaces, runtime stack, install commands, and proof gates. |
| `matdaemon_backend_status` | Inspect Python, platform, NumPy, and backend availability. |
| `matdaemon_validate_matrices` | Validate matrix payloads and return output shape or validation errors. |
| `matdaemon_matmul` | Multiply two matrices and return result plus timing metadata. |
| `matdaemon_similarity_top_k` | Compute normalized query/candidate embedding similarity and return top-k indexes and scores. |
| `matdaemon_use_cases` | Return all AI use cases or one use case by id. |
| `matdaemon_generate_api_payload` | Generate a ready-to-send HTTP API payload for sync or async matrix jobs. |
| `matdaemon_generate_github_action` | Generate a GitHub Actions snippet for benchmark runs. |
| `matdaemon_smoke_benchmark` | Run a bounded local square-matrix benchmark, capped at size 1024. |

## AI Workflows

### Coding agent setup

Use `matdaemon_platform_manifest` first so the coding agent knows which surfaces exist. Then use `matdaemon_backend_status` to choose `auto`, `numpy`, `tiled`, or `cuda`.

### Agent memory routing

Use `matdaemon_similarity_top_k` to rank memory embeddings for a query vector.

### Local RAG similarity

Use `matdaemon_similarity_top_k` to rank local document embeddings without requiring a vector database.

### Matrix job execution

Use `matdaemon_validate_matrices` before `matdaemon_matmul` when the AI client is constructing payloads dynamically.

### API handoff

Use `matdaemon_generate_api_payload` when a coding tool needs to call the HTTP mini platform instead of stdio MCP.

### CI handoff

Use `matdaemon_generate_github_action` to create a benchmark workflow snippet for releases or PR checks.

### Local proof

Use `matdaemon_smoke_benchmark` for a quick local runtime proof. It is intentionally capped to avoid runaway execution from tool callers.

## Security Posture

The MCP server exposes bounded matrix compute and discovery helpers only. It does not execute shell commands, read arbitrary files, mutate repositories, make network calls, or provide broad host access.

That is intentional: coding platforms can use MatDaemon as a compute/proof surface without receiving unsafe machine-control powers.

## Troubleshooting

### `pip install matdaemon` cannot find the package

The package has not been uploaded to PyPI yet. Use the GitHub/source install path until the first release is published.

### Windows ARM build failures from `cryptography` or `httptools`

MatDaemon no longer requires those packages for its normal dev, API, or MCP path. Use:

```bash
python -m pip install -e .[dev,api]
pytest -q
matdaemon mcp
```

The API extra uses plain `uvicorn`, not `uvicorn[standard]`, so `httptools` is not required. The MCP server is self-contained, so the external MCP dependency chain is not required.
