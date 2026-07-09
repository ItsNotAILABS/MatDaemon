"""Local RAG similarity example powered by MatDaemon."""

from __future__ import annotations

import numpy as np

import matdaemon as md


def top_k_documents(query_vectors: np.ndarray, document_vectors: np.ndarray, k: int = 3) -> np.ndarray:
    query_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8)
    doc_norm = document_vectors / (np.linalg.norm(document_vectors, axis=1, keepdims=True) + 1e-8)
    similarity = md.matmul(query_norm.astype(np.float32), doc_norm.astype(np.float32).T, backend="auto")
    return np.argsort(-similarity, axis=1)[:, :k]


def main() -> None:
    rng = np.random.default_rng(42)
    queries = rng.standard_normal((2, 1024), dtype=np.float32)
    documents = rng.standard_normal((50_000, 1024), dtype=np.float32)
    print(top_k_documents(queries, documents, k=3))


if __name__ == "__main__":
    main()
