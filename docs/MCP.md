# MatDaemon MCP Server

MatDaemon exposes matrix compute tools to MCP-compatible AI clients.

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

## Tools

### `matdaemon_matmul`

Multiply two matrices and return result plus timing metadata.

Input:

```json
{
  "a": [[1, 2], [3, 4]],
  "b": [[5, 6], [7, 8]],
  "backend": "auto",
  "dtype": "float32"
}
```

### `matdaemon_similarity_top_k`

Compute normalized embedding similarity and return top-k candidate indexes.

Input:

```json
{
  "queries": [[0.1, 0.2, 0.3]],
  "candidates": [[0.1, 0.2, 0.4], [0.9, 0.1, 0.1]],
  "k": 1,
  "backend": "auto"
}
```

### `matdaemon_use_cases`

Return the built-in AI use-case registry.

## Client Config Shape

For MCP clients that launch stdio servers, use this command shape:

```json
{
  "command": "matdaemon",
  "args": ["mcp"]
}
```

## Security Note

The MCP server only exposes bounded matrix compute helpers. It does not execute shell commands, read arbitrary files, or access network resources.
