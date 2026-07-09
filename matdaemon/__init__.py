"""Public MatDaemon SDK API."""

from .matdaemon import (
    BackendName,
    CudaUnavailableError,
    MatDaemon,
    MatrixHelperDaemon,
    MatrixResult,
    MatrixTask,
    MemoryPolicy,
    VectorizedMatrixMultiplier,
    cuda_available,
    matmul,
    register_signal_handler,
    resolve_backend,
    validate_matrices,
)

__version__ = "0.1.0"

__all__ = [
    "BackendName",
    "CudaUnavailableError",
    "MatDaemon",
    "MatrixHelperDaemon",
    "MatrixResult",
    "MatrixTask",
    "MemoryPolicy",
    "VectorizedMatrixMultiplier",
    "cuda_available",
    "matmul",
    "register_signal_handler",
    "resolve_backend",
    "validate_matrices",
]
