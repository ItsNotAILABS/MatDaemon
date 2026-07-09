"""AI use-case registry for MatDaemon platform surfaces."""

from __future__ import annotations

from typing import Optional

USE_CASES = [
    {"id": "agent-memory-routing", "title": "Agent memory routing", "description": "Score query embeddings against an agent memory matrix and select the highest-similarity memories.", "shape_pattern": "queries[M, D] @ memories[N, D].T -> scores[M, N]", "recommended_backend": "auto"},
    {"id": "local-rag-similarity", "title": "Local RAG similarity", "description": "Compute query-to-document similarity for local retrieval without a vector database dependency.", "shape_pattern": "queries[M, D] @ docs[N, D].T -> scores[M, N]", "recommended_backend": "auto"},
    {"id": "embedding-projection", "title": "Embedding projection", "description": "Project embeddings through a learned or generated weight matrix inside an AI pipeline.", "shape_pattern": "embeddings[B, Din] @ weights[Din, Dout] -> projected[B, Dout]", "recommended_backend": "numpy"},
    {"id": "attention-block", "title": "Attention-style score block", "description": "Compute QK-style score matrices for experiments, routing, or simulation layers.", "shape_pattern": "Q[T, D] @ K[S, D].T -> scores[T, S]", "recommended_backend": "tiled"},
    {"id": "simulation-worker", "title": "Simulation worker matrix step", "description": "Run repeated dense matrix steps in local scientific or agent-simulation workers.", "shape_pattern": "state[M, K] @ transition[K, N] -> next_state[M, N]", "recommended_backend": "auto"},
]


def get_use_case(use_case_id: str) -> Optional[dict]:
    for use_case in USE_CASES:
        if use_case["id"] == use_case_id:
            return use_case
    return None
