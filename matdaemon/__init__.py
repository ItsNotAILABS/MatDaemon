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
from .platform import PLATFORM_SURFACES, get_platform_manifest
from .text import hashing_embed, text_similarity_top_k

__version__ = "0.3.2"

__all__ = [
    "BackendName",
    "CudaUnavailableError",
    "MatDaemon",
    "MatrixHelperDaemon",
    "MatrixResult",
    "MatrixTask",
    "MemoryPolicy",
    "PLATFORM_SURFACES",
    "VectorizedMatrixMultiplier",
    "cuda_available",
    "get_platform_manifest",
    "hashing_embed",
    "matmul",
    "register_signal_handler",
    "resolve_backend",
    "text_similarity_top_k",
    "validate_matrices",
]
