"""MatDaemon SDK core."""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np

BackendName = Literal["auto", "numpy", "tiled", "cuda"]
TaskCallback = Callable[[str, Union[np.ndarray, Exception], float], None]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MatDaemon")


@dataclass(frozen=True)
class MemoryPolicy:
    """Controls when MatDaemon switches away from direct NumPy matmul."""

    max_direct_output_bytes: int = 1024 * 1024 * 1024
    small_block_size: int = 256
    medium_block_size: int = 512
    large_block_size: int = 1024
    medium_element_threshold: int = 10**6
    large_element_threshold: int = 10**7

    def block_size_for(self, matrix_shape_a: Tuple[int, int], matrix_shape_b: Tuple[int, int]) -> int:
        total_elements = matrix_shape_a[0] * matrix_shape_a[1] + matrix_shape_b[0] * matrix_shape_b[1]
        if total_elements > self.large_element_threshold:
            return self.large_block_size
        if total_elements > self.medium_element_threshold:
            return self.medium_block_size
        return self.small_block_size


@dataclass(frozen=True)
class MatrixTask:
    """A matrix multiplication job submitted to the daemon."""

    task_id: str
    matrix_a: np.ndarray
    matrix_b: np.ndarray
    backend: BackendName = "auto"
    callback: Optional[TaskCallback] = None
    creation_time: float = field(default_factory=time.time)

    def validate(self) -> None:
        validate_matrices(self.matrix_a, self.matrix_b)


@dataclass(frozen=True)
class MatrixResult:
    """Result metadata returned by daemon jobs."""

    task_id: str
    result: np.ndarray
    duration_seconds: float
    backend: str
    output_shape: Tuple[int, int]


class CudaUnavailableError(RuntimeError):
    """Raised when the CUDA backend is requested without usable CuPy/CUDA."""


def validate_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray) -> None:
    if not isinstance(matrix_a, np.ndarray) or not isinstance(matrix_b, np.ndarray):
        raise TypeError("Both inputs must be numpy.ndarray instances.")
    if matrix_a.ndim != 2 or matrix_b.ndim != 2:
        raise ValueError(f"Matrices must be 2D. Matrix A: {matrix_a.ndim}D, Matrix B: {matrix_b.ndim}D.")
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError(
            "Dimension mismatch for multiplication: "
            f"Matrix A columns ({matrix_a.shape[1]}) must match Matrix B rows ({matrix_b.shape[0]})."
        )


def _as_contiguous(matrix: np.ndarray) -> np.ndarray:
    return matrix if matrix.flags["C_CONTIGUOUS"] else np.ascontiguousarray(matrix)


class VectorizedMatrixMultiplier:
    """CPU matrix multiplier with direct NumPy and memory-aware tiled modes."""

    def __init__(self, memory_policy: Optional[MemoryPolicy] = None):
        self.memory_policy = memory_policy or MemoryPolicy()

    def get_optimal_block_size(self, matrix_shape_a: Tuple[int, int], matrix_shape_b: Tuple[int, int]) -> int:
        return self.memory_policy.block_size_for(matrix_shape_a, matrix_shape_b)

    def multiply(self, A: np.ndarray, B: np.ndarray, force_tiled: bool = False) -> np.ndarray:
        validate_matrices(A, B)
        A = _as_contiguous(A)
        B = _as_contiguous(B)
        m, n = A.shape
        _, p = B.shape
        estimated_output_bytes = m * p * A.dtype.itemsize

        if estimated_output_bytes < self.memory_policy.max_direct_output_bytes and not force_tiled:
            return np.matmul(A, B)

        block_size = self.get_optimal_block_size(A.shape, B.shape)
        C = np.zeros((m, p), dtype=np.result_type(A.dtype, B.dtype))
        for i in range(0, m, block_size):
            i_end = min(i + block_size, m)
            for k in range(0, n, block_size):
                k_end = min(k + block_size, n)
                A_tile = A[i:i_end, k:k_end]
                for j in range(0, p, block_size):
                    j_end = min(j + block_size, p)
                    C[i:i_end, j:j_end] += A_tile @ B[k:k_end, j:j_end]
        return C


class NumpyBackend:
    name = "numpy"

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        validate_matrices(A, B)
        return np.matmul(A, B)


class TiledBackend:
    name = "tiled"

    def __init__(self, memory_policy: Optional[MemoryPolicy] = None):
        self.multiplier = VectorizedMatrixMultiplier(memory_policy)

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self.multiplier.multiply(A, B, force_tiled=True)


class CUDABackend:
    name = "cuda"

    def __init__(self, tile_size: int = 128):
        try:
            from .backends.cuda_backend import CUDABackend as CUDAImpl
            self._impl = CUDAImpl(tile_size=tile_size)
        except Exception as exc:  # pragma: no cover - depends on optional CuPy/CUDA install
            raise CudaUnavailableError(
                "CUDA backend requires CuPy and a working CUDA runtime. "
                "Install `matdaemon[cuda]` or a CuPy build that matches your CUDA version."
            ) from exc

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        validate_matrices(A, B)
        return self._impl.matmul(A, B)


def cuda_available() -> bool:
    try:
        CUDABackend()
        return True
    except Exception:
        return False


def resolve_backend(
    backend: BackendName = "auto",
    memory_policy: Optional[MemoryPolicy] = None,
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
):
    if backend == "numpy":
        return NumpyBackend()
    if backend == "tiled":
        return TiledBackend(memory_policy)
    if backend == "cuda":
        return CUDABackend()
    if backend != "auto":
        raise ValueError(f"Unsupported backend: {backend}")

    if cuda_available():
        return CUDABackend()

    if A is not None and B is not None:
        policy = memory_policy or MemoryPolicy()
        estimated_output_bytes = A.shape[0] * B.shape[1] * A.dtype.itemsize
        if estimated_output_bytes >= policy.max_direct_output_bytes:
            return TiledBackend(policy)
    return NumpyBackend()


def matmul(
    A: np.ndarray,
    B: np.ndarray,
    backend: BackendName = "auto",
    memory_policy: Optional[MemoryPolicy] = None,
) -> np.ndarray:
    """Multiply two matrices with an explicit or automatically selected backend."""

    validate_matrices(A, B)
    selected_backend = resolve_backend(backend=backend, memory_policy=memory_policy, A=A, B=B)
    return selected_backend.matmul(A, B)


class MatDaemon:
    """High-level SDK object for synchronous and asynchronous matrix jobs."""

    def __init__(
        self,
        backend: BackendName = "auto",
        max_queue_size: int = 100,
        memory_policy: Optional[MemoryPolicy] = None,
    ):
        self.backend = backend
        self.memory_policy = memory_policy or MemoryPolicy()
        self._task_queue: queue.Queue[Optional[MatrixTask]] = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._active_tasks: Dict[str, float] = {}
        self._results: Dict[str, Union[MatrixResult, Exception]] = {}
        self._lock = threading.Lock()

    def __enter__(self) -> "MatDaemon":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.shutdown()

    def matmul(self, A: np.ndarray, B: np.ndarray, backend: Optional[BackendName] = None) -> np.ndarray:
        return matmul(A, B, backend=backend or self.backend, memory_policy=self.memory_policy)

    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Daemon is already running.")
            return
        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, name="MatDaemonWorker", daemon=True)
        self._worker_thread.start()

    def submit(self, task: MatrixTask) -> bool:
        if self._shutdown_event.is_set():
            return False
        try:
            task.validate()
            self._task_queue.put(task, block=False)
        except (ValueError, TypeError, queue.Full):
            return False
        with self._lock:
            self._active_tasks[task.task_id] = time.time()
        return True

    def submit_matmul(
        self,
        A: np.ndarray,
        B: np.ndarray,
        task_id: Optional[str] = None,
        backend: Optional[BackendName] = None,
        callback: Optional[TaskCallback] = None,
    ) -> str:
        task = MatrixTask(
            task_id=task_id or str(uuid.uuid4()),
            matrix_a=A,
            matrix_b=B,
            backend=backend or self.backend,
            callback=callback,
        )
        if not self.submit(task):
            raise RuntimeError(f"Unable to submit task {task.task_id}")
        return task.task_id

    def result(self, task_id: str) -> Optional[Union[MatrixResult, Exception]]:
        with self._lock:
            return self._results.get(task_id)

    def active_tasks(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._active_tasks)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._shutdown_event.set()
        try:
            self._task_queue.put(None, block=False)
        except queue.Full:
            try:
                self._task_queue.get_nowait()
                self._task_queue.put(None, block=False)
            except Exception:
                pass
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        while not self._shutdown_event.is_set() or not self._task_queue.empty():
            try:
                task = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                self._task_queue.task_done()
                break
            start_time = time.perf_counter()
            try:
                selected_backend = resolve_backend(task.backend, self.memory_policy, task.matrix_a, task.matrix_b)
                output = selected_backend.matmul(task.matrix_a, task.matrix_b)
                duration = time.perf_counter() - start_time
                result = MatrixResult(task.task_id, output, duration, selected_backend.name, output.shape)
                with self._lock:
                    self._results[task.task_id] = result
                if task.callback:
                    task.callback(task.task_id, output, duration)
            except Exception as exc:
                duration = time.perf_counter() - start_time
                with self._lock:
                    self._results[task.task_id] = exc
                if task.callback:
                    task.callback(task.task_id, exc, duration)
            finally:
                with self._lock:
                    self._active_tasks.pop(task.task_id, None)
                self._task_queue.task_done()


MatrixHelperDaemon = MatDaemon


def register_signal_handler(daemon: MatDaemon) -> None:
    """Hook standard termination signals for graceful shutdown."""

    def handler(signum: int, frame: Any) -> None:
        logger.info("Captured signal %s. System exiting...", signal.Signals(signum).name)
        daemon.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


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
