"""Deterministic, dependency-free text vectorization for MatDaemon.

Turns strings into dense float vectors using the signed hashing trick
(feature hashing). This keeps MatDaemon's safety boundary intact: pure NumPy
compute, no model download, no network, no file reads -- and fully
reproducible across processes and platforms, unlike Python's builtin ``hash``,
which is salted per-process (``PYTHONHASHSEED``) and therefore unusable for a
vectorizer whose output must be stable.

The vectors are lexical, not neural: similarity reflects shared words and
(optionally) shared character n-grams, not learned semantics. Feed the vectors
straight into ``similarity_top_k`` for ranking, or swap in your own neural
embeddings when you need semantic matching -- MatDaemon runs the same top-k
math either way. This module is the "text in, vectors out" front door so the
compute surface is usable on raw strings without a separate embedding step.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .matdaemon import matmul


def _stable_hash(token: str) -> int:
    """Process- and platform-stable 64-bit token hash (blake2b).

    Deliberately not Python's builtin ``hash``: that is salted per process, so
    two runs would produce different vectors for the same text and break any
    cached-embedding or cross-service workflow.
    """
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def _tokens(text: str, word_ngram: int, char_ngram: int) -> list[str]:
    lowered = text.lower()
    words = lowered.split()
    tokens: list[str] = []
    for n in range(1, max(1, word_ngram) + 1):
        for i in range(len(words) - n + 1):
            tokens.append("w:" + " ".join(words[i : i + n]))
    if char_ngram and char_ngram > 0:
        squashed = " ".join(words)
        for i in range(len(squashed) - char_ngram + 1):
            tokens.append("c:" + squashed[i : i + char_ngram])
    return tokens


def hashing_embed(
    texts: Sequence[str],
    dim: int = 256,
    word_ngram: int = 1,
    char_ngram: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """Embed ``texts`` into a ``(len(texts), dim)`` float32 matrix.

    Uses the signed hashing trick: each token maps to a bucket and a +/-1 sign,
    both derived from a stable hash. Shared tokens reinforce constructively
    while hash collisions between distinct tokens cancel in expectation, which
    keeps the inner-product (cosine) estimate unbiased.

    Args:
        texts: strings to embed.
        dim: output dimensionality (number of hash buckets). Larger = fewer
            collisions, more memory. 256 is a sensible default for short text.
        word_ngram: include word n-grams up to this length (1 = unigrams).
        char_ngram: if > 0, also include character n-grams of this length --
            useful for morphological/typo robustness ("launch" ~ "launching").
        normalize: L2-normalize each row so the dot product is cosine similarity.

    Returns:
        float32 array of shape ``(len(texts), dim)``.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if word_ngram < 1:
        raise ValueError("word_ngram must be >= 1")
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in _tokens(str(text), word_ngram, char_ngram):
            digest = _stable_hash(token)
            index = digest % dim
            sign = 1.0 if (digest >> 63) & 1 == 0 else -1.0
            matrix[row, index] += sign
    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
    return matrix


def text_similarity_top_k(
    queries: Sequence[str],
    candidates: Sequence[str],
    k: int = 5,
    dim: int = 256,
    word_ngram: int = 1,
    char_ngram: int = 0,
    backend: str = "auto",
) -> dict:
    """Rank ``candidates`` against ``queries`` by lexical cosine similarity.

    Composes :func:`hashing_embed` with the same normalized-matmul + top-k that
    powers ``similarity_top_k``, so callers can rank raw strings in one call.

    Returns a dict with ``top_k`` (candidate indexes per query), ``top_scores``,
    ``scores_shape``, ``dim``, and ``backend``.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    q = hashing_embed(queries, dim=dim, word_ngram=word_ngram, char_ngram=char_ngram, normalize=True)
    c = hashing_embed(candidates, dim=dim, word_ngram=word_ngram, char_ngram=char_ngram, normalize=True)
    scores = matmul(q, c.T, backend=backend)
    k = max(1, min(k, len(candidates)))
    top_k = np.argsort(-scores, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top_k, axis=1)
    return {
        "top_k": top_k.tolist(),
        "top_scores": top_scores.tolist(),
        "scores_shape": list(scores.shape),
        "dim": dim,
        "backend": backend,
    }
