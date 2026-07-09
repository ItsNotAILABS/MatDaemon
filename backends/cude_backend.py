"""Compatibility shim for the original misspelled CUDA backend filename.

Use `backends.cuda_backend` for new code.
"""

from .cuda_backend import CUDABackend, GEMM_KERNEL

__all__ = ["CUDABackend", "GEMM_KERNEL"]
