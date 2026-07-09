from typing import Optional, Callable, Union
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from .backends.numpy_backend import NumPyBackend
from .backends.cuda_backend import CUDABackend


class MatDaemon:
    """High-performance matrix multiplication SDK with CPU and CUDA backends."""

    def __init__(self, backend: str = "auto", tile_size: int = 128):
        self.backend_name = backend
        self.tile_size = tile_size

        if backend == "cuda" or (backend == "auto" and HAS_CUPY):
            if not HAS_CUPY:
                raise ImportError(
                    "CuPy is required for CUDA backend.\n"
                    "Install with: pip install cupy-cuda12x  (or cupy-cuda11x)"
                )
            self.backend = CUDABackend(tile_size=tile_size)
        else:
            self.backend = NumPyBackend()

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self.backend.matmul(A, B)

    def submit(self, A: np.ndarray, B: np.ndarray, callback: Optional[Callable] = None):
        result = self.matmul(A, B)
        if callback:
            callback(result)
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        pass  # Extend later for async cleanup


def matmul(A: np.ndarray, B: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Convenience function for one-off matrix multiplication."""
    with MatDaemon(backend=backend) as daemon:
        return daemon.matmul(A, B)
