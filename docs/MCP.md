# MatDaemon MCP Server

MatDaemon exposes bounded matrix compute and platform-discovery tools to MCP-compatible AI clients.

## Install

```bash
pip install "matdaemon[mcp]"
```

From source:

```bash
python -m pip install -e .[mcp]
```

## Run

```bash
matdaemon mcp
```

## Client Config Shape

```json
{
  "command": "matdaemon",
  "args": ["mcp"]
}
```

## Tools

| Tool | Purpose |
| --- | --- |
| `matdaemon_matmul` | multiply two matrices and return result plus timing metadata |
| `matdaemon_similarity_top_k` | compute normalized query/candidate embedding similarity and return top-k indexes |
| `matdaemon_use_cases` | return the built-in AI use-case registry |
| `matdaemon_platform_manifest` | return the MatDaemon product surfaces, runtime stack, install commands, and proof gates |

## AI Workflows

### Agent memory routing

Use `matdaemon_similarity_top_k` to rank memory embeddings for a query vector.

### Local RAG similarity

Use `matdaemon_similarity_top_k` to rank local document embeddings without requiring a vector database.

### Matrix job execution

Use `matdaemon_matmul` when the AI client needs direct dense matrix multiplication for projection, scoring, simulation, or routing.

### Tool discovery

Use `matdaemon_platform_manifest` when an AI client needs to understand which MatDaemon surfaces exist before choosing SDK, API, MCP, GitHub Action, or CUDA paths.

## Security Note

The MCP server only exposes bounded matrix compute and discovery helpers. It does not execute shell commands, read arbitrary files, mutate repositories, or access network resources.
