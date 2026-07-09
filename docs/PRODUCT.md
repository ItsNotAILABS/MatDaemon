# MatDaemon Product Surface

MatDaemon is a shippable Python SDK and compute daemon for memory-aware matrix multiplication in agentic AI, ML, simulation, and automation workflows.

## Product Surfaces

- **SDK:** `import matdaemon as md` and call `md.matmul(A, B)`.
- **Daemon:** use `MatDaemon` as a context-managed async worker for queued jobs.
- **CLI:** run `.npy` matrix multiplication and quick benchmarks from a terminal.
- **Backends:** choose `auto`, `numpy`, `tiled`, or optional `cuda`.
- **Proof Layer:** tests and benchmark harness establish correctness against NumPy.

## Ship-Day Positioning

MatDaemon is not a general ML framework. It is a focused compute substrate for teams that need matrix multiplication to be easy to embed, easy to benchmark, and harder to crash under large outputs.

## First Customer Flow

1. Install `matdaemon`.
2. Multiply matrices through the SDK or CLI.
3. Use `backend="tiled"` when memory safety matters.
4. Use `backend="cuda"` when CuPy and CUDA are available.
5. Run benchmarks to produce machine-specific proof.

## Next Product Gates

- HTTP service wrapper for remote jobs.
- Result artifact storage for long-running jobs.
- Streaming status and cancellation.
- Published benchmark table with hardware notes.
