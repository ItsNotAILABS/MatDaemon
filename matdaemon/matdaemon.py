"""
MatDaemon
High-Performance Asynchronous Vectorized Matrix Multiplication Daemon
for AI, agentic systems, and large-scale computations.
"""

import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MatDaemon")


@dataclass(frozen=True)
class MatrixTask:
    """Represents a matrix multiplication job submitted to the daemon."""
    task_id: str
    matrix_a: np.ndarray
    matrix_b: np.ndarray
    callback: Optional[Callable[[str, Union[np.ndarray, Exception], float], None]] = None
    creation_time: float = field(default_factory=time.time)

    def validate(self) -> None:
        """Validates matrices for dimension compatibility and data integrity."""
        if not isinstance(self.matrix_a, np.ndarray) or not isinstance(self.matrix_b, np.ndarray):
            raise TypeError("Both inputs must be of type numpy.ndarray.")

        if self.matrix_a.ndim != 2 or self.matrix_b.ndim != 2:
            raise ValueError(
                f"Matrices must be 2D. "
                f"Matrix A: {self.matrix_a.ndim}D, Matrix B: {self.matrix_b.ndim}D."
            )

        if self.matrix_a.shape[1] != self.matrix_b.shape[0]:
            raise ValueError(
                f"Dimension mismatch for multiplication: "
                f"Matrix A columns ({self.matrix_a.shape[1]}) must match "
                f"Matrix B rows ({self.matrix_b.shape[0]})."
            )


class VectorizedMatrixMultiplier:
    """
    Handles optimized, high-performance vectorized matrix multiplication.
    Uses tiling to keep memory usage bounded, preventing OOM errors.
    """

    @staticmethod
    def get_optimal_block_size(matrix_shape_a: Tuple[int, int], matrix_shape_b: Tuple[int, int]) -> int:
        """Dynamically calculates optimal block size for tiled multiplication."""
        total_elements = (matrix_shape_a[0] * matrix_shape_a[1]) + (matrix_shape_b[0] * matrix_shape_b[1])
        if total_elements > 10**7:
            return 1024
        elif total_elements > 10**6:
            return 512
        else:
            return 256

    @classmethod
    def multiply(cls, A: np.ndarray, B: np.ndarray, force_tiled: bool = False) -> np.ndarray:
        """Executes highly optimized matrix multiplication with automatic tiling for large arrays."""
        if not A.flags['C_CONTIGUOUS']:
            A = np.ascontiguousarray(A)
        if not B.flags['C_CONTIGUOUS']:
            B = np.ascontiguousarray(B)

        m, n = A.shape
        _, p = B.shape

        estimated_output_bytes = m * p * A.itemsize
        memory_threshold_bytes = 1024 * 1024 * 1024  # 1 GB

        if estimated_output_bytes < memory_threshold_bytes and not force_tiled:
            return np.matmul(A, B)

        logger.info(f"Using Tiled Vectorized multiplication for target output size: {estimated_output_bytes / (1024**2):.2f} MB")
        block_size = cls.get_optimal_block_size(A.shape, B.shape)
        C = np.zeros((m, p), dtype=A.dtype)

        for i in range(0, m, block_size):
            i_end = min(i + block_size, m)
            for k in range(0, n, block_size):
                k_end = min(k + block_size, n)
                A_tile = A[i:i_end, k:k_end]
                for j in range(0, p, block_size):
                    j_end = min(j + block_size, p)
                    C[i:i_end, j:j_end] += A_tile @ B[k:k_end, j:j_end]

        return C


class MatrixHelperDaemon:
    """
    Background daemon that processes matrix multiplication tasks asynchronously
    with memory-safe tiled execution and robust error handling.
    """

    def __init__(self, max_queue_size: int = 100):
        self._task_queue: queue.Queue[Optional[MatrixTask]] = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._active_tasks: Dict[str, float] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Daemon is already running.")
            return

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="MatDaemonWorker",
            daemon=True
        )
        self._worker_thread.start()
        logger.info("MatDaemon started successfully.")

    def submit(self, task: MatrixTask) -> bool:
        if self._shutdown_event.is_set():
            logger.error(f"Cannot submit task '{task.task_id}'; daemon is shutting down.")
            return False

        try:
            task.validate()
        except (ValueError, TypeError) as e:
            logger.error(f"Task validation failed for '{task.task_id}': {e}")
            return False

        try:
            self._task_queue.put(task, block=False)
            with self._lock:
                self._active_tasks[task.task_id] = time.time()
            logger.debug(f"Task '{task.task_id}' queued successfully.")
            return True
        except queue.Full:
            logger.warning(f"Task queue full. Rejected task: '{task.task_id}'")
            return False

    def shutdown(self, timeout: float = 5.0) -> None:
        logger.info("Shutdown requested. Gracefully stopping daemon thread...")
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
            if self._worker_thread.is_alive():
                logger.warning("Daemon thread did not terminate within timeout; forcing exit.")
            else:
                logger.info("Daemon thread shut down cleanly.")

    def _run_loop(self) -> None:
        while not self._shutdown_event.is_set() or not self._task_queue.empty():
            try:
                task = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:
                self._task_queue.task_done()
                break

            logger.info(f"Processing task '{task.task_id}' (Shape: {task.matrix_a.shape} x {task.matrix_b.shape})")
            start_time = time.perf_counter()

            try:
                result = VectorizedMatrixMultiplier.multiply(task.matrix_a, task.matrix_b)
                duration = time.perf_counter() - start_time
                logger.info(f"Completed task '{task.task_id}' in {duration:.4f} seconds.")

                if task.callback:
                    try:
                        task.callback(task.task_id, result, duration)
                    except Exception as cb_err:
                        logger.error(f"Error executing callback for '{task.task_id}': {cb_err}", exc_info=True)

            except Exception as exc:
                duration = time.perf_counter() - start_time
                logger.error(f"Computation failure on task '{task.task_id}' after {duration:.4f}s: {exc}", exc_info=True)
                if task.callback:
                    try:
                        task.callback(task.task_id, exc, duration)
                    except Exception as cb_err:
                        logger.error(f"Error executing error callback for '{task.task_id}': {cb_err}")
            finally:
                with self._lock:
                    self._active_tasks.pop(task.task_id, None)
                self._task_queue.task_done()


def register_signal_handler(daemon: MatrixHelperDaemon) -> None:
    """Hooks standard termination signals for graceful shutdown."""
    def handler(signum: int, frame: Any) -> None:
        logger.info(f"Captured signal {signal.Signals(signum).name}. System exiting...")
        daemon.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


if __name__ == "__main__":
    logger.info("Initializing MatDaemon performance verification...")

    results_received = {}

    def sample_callback(task_id: str, payload: Union[np.ndarray, Exception], elapsed_time: float) -> None:
        if isinstance(payload, Exception):
            logger.error(f"Callback failure for Task {task_id}: {payload}")
        else:
            results_received[task_id] = payload
            logger.info(f"Callback successful. Shape: {payload.shape}. Time: {elapsed_time:.4f}s")

    daemon = MatrixHelperDaemon()
    register_signal_handler(daemon)
    daemon.start()

    # Example tasks
    m_a_1 = np.random.randn(1000, 1200).astype(np.float32)
    m_b_1 = np.random.randn(1200, 1500).astype(np.float32)
    m_a_2 = np.random.randn(3000, 3000).astype(np.float32)
    m_b_2 = np.random.randn(3000, 3000).astype(np.float32)

    task_1 = MatrixTask("Standard_Vectorized_01", m_a_1, m_b_1, sample_callback)
    task_2 = MatrixTask("Tiled_MemorySafe_02", m_a_2, m_b_2, sample_callback)

    daemon.submit(task_1)
    daemon.submit(task_2)

    # Wait for completion
    max_wait = 30
    waited = 0
    while len(results_received) < 2 and waited < max_wait:
        time.sleep(0.5)
        waited += 0.5

    if "Standard_Vectorized_01" in results_received:
        res = results_received["Standard_Vectorized_01"]
        expected = (1000, 1500)
        assert res.shape == expected, "Output dimension mismatch!"
        logger.info(f"Verification passed: Output shape {res.shape} correct.")
    else:
        logger.error("Verification failed: Did not receive expected results.")

    daemon.shutdown()
