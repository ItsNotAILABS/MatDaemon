"""Backend contract for matrix multiplication providers."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BaseBackend(Protocol):
    name: str

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiply A and B and return a NumPy array."""
        ...
