"""Memory-aware tiled CPU backend."""

from __future__ import annotations

import numpy as np

from ..matdaemon import MemoryPolicy, VectorizedMatrixMultiplier


class TiledBackend:
    name = "tiled"

    def __init__(self, memory_policy: MemoryPolicy | None = None):
        self.multiplier = VectorizedMatrixMultiplier(memory_policy)

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self.multiplier.multiply(A, B, force_tiled=True)
