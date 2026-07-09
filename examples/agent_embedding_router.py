"""AI-native MatDaemon example: route agent memory by embedding similarity.

This example simulates an agent with a bank of memory embeddings. MatDaemon
computes query-to-memory scores with one matrix multiplication, then returns the
top memories to route into the next agent step.

Run:
    python examples/agent_embedding_router.py
"""

from __future__ import annotations

import numpy as np

import matdaemon as md


def normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / denom


def route_memories(query_embeddings: np.ndarray, memory_embeddings: np.ndarray, top_k: int = 5) -> np.ndarray:
    query_embeddings = normalize(query_embeddings.astype(np.float32))
    memory_embeddings = normalize(memory_embeddings.astype(np.float32))
    scores = md.matmul(query_embeddings, memory_embeddings.T, backend="auto")
    return np.argsort(-scores, axis=1)[:, :top_k]


def main() -> None:
    rng = np.random.default_rng(7)
    queries = rng.standard_normal((4, 768), dtype=np.float32)
    memories = rng.standard_normal((10_000, 768), dtype=np.float32)
    top_memory_ids = route_memories(queries, memories, top_k=5)
    print("Top memory IDs per agent query:")
    print(top_memory_ids)


if __name__ == "__main__":
    main()
