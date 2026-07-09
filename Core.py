import numpy as np
from contextlib import contextmanager
from typing import Optional, Callable, Union

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from .backends.numpy_backend import NumPyBackend
from .backends.cuda_backend import CUDABackend


class MatDaemon:
    def __init__(self, backend: str = "auto", tile_size: int = 128):
        self.backend_name = backend
        self.tile_size = tile_size

        if backend == "cuda" or (backend == "auto" and HAS_CUPY):
            if not HAS_CUPY:
                raise ImportError("CuPy is required for CUDA backend. Install with: pip install cupy-cuda12x")
            self.backend = CUDABackend(tile_size=tile_size)
        else:
            self.backend = NumPyBackend()

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """High-level matrix multiplication (synchronous for now)."""
        return self.backend.matmul(A, B)

    def submit(self, A: np.ndarray, B: np.ndarray, callback: Optional[Callable] = None):
        """Async-style submission (background thread in future versions)."""
        result = self.matmul(A, B)
        if callback:
            callback(result)
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        if hasattr(self.backend, "shutdown"):
            self.backend.shutdown()


# Convenience function
def matmul(A, B, backend="auto"):
    with MatDaemon(backend=backend) as daemon:
        return daemon.matmul(A, B)
