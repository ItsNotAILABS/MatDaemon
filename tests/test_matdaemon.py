import time

import numpy as np
import pytest

import matdaemon as md


def test_public_matmul_matches_numpy():
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    B = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)

    result = md.matmul(A, B, backend="numpy")

    np.testing.assert_allclose(result, np.matmul(A, B))


def test_tiled_backend_matches_numpy():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((33, 17), dtype=np.float32)
    B = rng.standard_normal((17, 21), dtype=np.float32)

    result = md.matmul(A, B, backend="tiled")

    np.testing.assert_allclose(result, np.matmul(A, B), rtol=1e-5, atol=1e-5)


def test_validation_rejects_bad_shapes():
    A = np.ones((2, 3), dtype=np.float32)
    B = np.ones((4, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        md.matmul(A, B, backend="numpy")


def test_daemon_submit_matmul_stores_result():
    A = np.eye(3, dtype=np.float32)
    B = np.ones((3, 2), dtype=np.float32)

    with md.MatDaemon(backend="numpy") as daemon:
        task_id = daemon.submit_matmul(A, B)
        deadline = time.time() + 5
        result = None
        while time.time() < deadline:
            result = daemon.result(task_id)
            if result is not None:
                break
            time.sleep(0.05)

    assert isinstance(result, md.MatrixResult)
    np.testing.assert_allclose(result.result, np.matmul(A, B))
    assert result.backend == "numpy"


def test_cuda_backend_fails_cleanly_when_unavailable():
    A = np.eye(2, dtype=np.float32)
    B = np.eye(2, dtype=np.float32)
    try:
        result = md.matmul(A, B, backend="cuda")
    except md.CudaUnavailableError as exc:
        assert "CUDA backend requires CuPy" in str(exc)
    else:
        assert result.shape == (2, 2)
