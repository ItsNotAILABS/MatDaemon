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

### `matdaemon_similarity_top_k`

Compute normalized embedding similarity and return top-k candidate indexes.

### `matdaemon_use_cases`

Return the built-in AI use-case registry.

## Client Config Shape

```json
{
  "command": "matdaemon",
  "args": ["mcp"]
}
```

## Security Note

The MCP server only exposes bounded matrix compute helpers. It does not execute shell commands, read arbitrary files, or access network resources.
