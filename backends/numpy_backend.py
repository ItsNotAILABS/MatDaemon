import numpy as np
from .base import BaseBackend


class NumPyBackend(BaseBackend):
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if not isinstance(A, np.ndarray):
            A = np.asarray(A)
        if not isinstance(B, np.ndarray):
            B = np.asarray(B)
        return np.matmul(A, B)
