"""NumPy backend."""

from __future__ import annotations

import numpy as np

from ..matdaemon import validate_matrices


class NumpyBackend:
    name = "numpy"

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        validate_matrices(A, B)
        return np.matmul(A, B)
