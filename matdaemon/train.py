"""Gradient-descent model training on the MatDaemon matmul backend.

A model's training loop is dense linear algebra: the forward pass is
``X @ W``, and the backward pass is ``X.T @ error`` and ``H.T @ delta`` --
all matmuls. This module trains real models (logistic regression and a
one-hidden-layer MLP) with every forward and backward matrix product routed
through :func:`matdaemon.matmul`, so MatDaemon is the training compute engine,
not just an inference primitive.

Everything is pure NumPy: no autograd framework, no downloaded weights, no
network, no file reads. Datasets are generated deterministically in-process
for demos and tests, or you pass your own ``(X, y)`` arrays. Training is
seedable and reproducible.

The backprop is verified two ways in the test suite: (1) trained models hit
the expected accuracy on separable / XOR data, and (2) the analytic gradient
matches a finite-difference numerical gradient -- the standard proof that a
hand-written backward pass is correct.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .matdaemon import matmul

# Bounds keep the training tools a safe, capped compute surface (mirroring the
# smoke-benchmark size cap) when driven from the MCP / HTTP tool API.
MAX_SAMPLES = 100_000
MAX_FEATURES = 4_096
MAX_EPOCHS = 20_000
MAX_HIDDEN = 1_024


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))),
                    np.exp(np.clip(z, -60, 60)) / (1.0 + np.exp(np.clip(z, -60, 60))))


def binary_cross_entropy(prob: np.ndarray, target: np.ndarray) -> float:
    """Mean binary cross-entropy with clipping for numerical stability."""
    p = np.clip(prob, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(target * np.log(p) + (1.0 - target) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# Deterministic in-process datasets (no file/network access)
# ---------------------------------------------------------------------------


def make_blobs_binary(n: int = 200, dim: int = 2, seed: int = 0, separation: float = 2.5):
    """Two Gaussian blobs -- a linearly separable binary classification set."""
    rng = np.random.default_rng(seed)
    half = n // 2
    centre = np.zeros(dim)
    centre[0] = separation
    X0 = rng.standard_normal((half, dim)) - centre
    X1 = rng.standard_normal((n - half, dim)) + centre
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(half), np.ones(n - half)]).reshape(-1, 1)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def make_xor(n: int = 400, seed: int = 0, noise: float = 0.15):
    """Noisy XOR -- not linearly separable; needs a hidden layer to solve."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    label = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float64)
    X = X + rng.normal(0.0, noise, size=X.shape)
    return X, label.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Logistic regression (convex; forward X@W, gradient X.T@error)
# ---------------------------------------------------------------------------


def train_logistic_regression(
    X, y, epochs: int = 300, lr: float = 0.1, l2: float = 0.0, seed: int = 0, backend: str = "auto"
) -> dict:
    """Train binary logistic regression by full-batch gradient descent.

    Forward: ``p = sigmoid(X @ W + b)``. Gradient: ``dW = X.T @ (p - y) / n``.
    Both the forward and the gradient are matmuls through the chosen backend.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n, d = X.shape
    W = np.zeros((d, 1), dtype=np.float64)  # convex problem: zero init is fine
    b = 0.0
    loss_history: list[float] = []
    for _ in range(epochs):
        p = sigmoid(matmul(X, W, backend=backend) + b)
        error = p - y
        grad_W = matmul(X.T, error, backend=backend) / n + l2 * W
        grad_b = float(np.mean(error))
        W -= lr * grad_W
        b -= lr * grad_b
        loss_history.append(binary_cross_entropy(p, y))
    prob = sigmoid(matmul(X, W, backend=backend) + b)
    accuracy = float(np.mean((prob >= 0.5).astype(np.float64) == y))
    return {
        "model": "logistic_regression",
        "weights": W.ravel().tolist(),
        "bias": b,
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "accuracy": accuracy,
        "epochs": epochs,
    }


def predict_logistic(X, weights, bias) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    return sigmoid(matmul(X, W) + bias).ravel()


# ---------------------------------------------------------------------------
# One-hidden-layer MLP (ReLU) -- learns non-linear boundaries like XOR
# ---------------------------------------------------------------------------


def train_mlp(
    X, y, hidden: int = 8, epochs: int = 2000, lr: float = 0.1, seed: int = 0, backend: str = "auto"
) -> dict:
    """Train a 1-hidden-layer ReLU MLP for binary classification.

    Forward: ``H = relu(X @ W1 + b1)``, ``p = sigmoid(H @ W2 + b2)``.
    Backprop is a chain of matmuls (``H.T @ dz``, ``dz @ W2.T``, ``X.T @ dH``).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)  # He init
    b1 = np.zeros((1, hidden))
    W2 = rng.standard_normal((hidden, 1)) * np.sqrt(2.0 / hidden)
    b2 = 0.0
    loss_history: list[float] = []
    for _ in range(epochs):
        pre = matmul(X, W1, backend=backend) + b1
        H = np.maximum(pre, 0.0)
        p = sigmoid(matmul(H, W2, backend=backend) + b2)
        loss_history.append(binary_cross_entropy(p, y))

        dz = (p - y) / n                                   # (n,1)
        dW2 = matmul(H.T, dz, backend=backend)             # (h,1)
        db2 = float(np.sum(dz))
        dH = matmul(dz, W2.T, backend=backend)             # (n,h)
        dpre = dH * (pre > 0.0)                             # ReLU grad
        dW1 = matmul(X.T, dpre, backend=backend)           # (d,h)
        db1 = np.sum(dpre, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    H = np.maximum(matmul(X, W1, backend=backend) + b1, 0.0)
    prob = sigmoid(matmul(H, W2, backend=backend) + b2)
    accuracy = float(np.mean((prob >= 0.5).astype(np.float64) == y))
    return {
        "model": "mlp",
        "hidden": hidden,
        "params": {"W1": W1.tolist(), "b1": b1.tolist(), "W2": W2.tolist(), "b2": b2},
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "accuracy": accuracy,
        "epochs": epochs,
    }


# ---------------------------------------------------------------------------
# Tool entry point (bounded)
# ---------------------------------------------------------------------------


def train_classifier(
    features,
    labels,
    model: str = "logistic_regression",
    epochs: int = 300,
    lr: float = 0.1,
    hidden: int = 8,
    seed: int = 0,
    backend: str = "auto",
) -> dict:
    """Bounded classifier training entry point for the MCP / HTTP tool surface.

    Returns learned parameters, the loss curve, and training accuracy. Sizes
    are capped so this stays a safe compute surface.
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("features must be a 2D array")
    if X.shape[0] > MAX_SAMPLES or X.shape[1] > MAX_FEATURES:
        raise ValueError(f"dataset exceeds bounds ({MAX_SAMPLES} samples x {MAX_FEATURES} features)")
    epochs = max(1, min(int(epochs), MAX_EPOCHS))
    if model == "logistic_regression":
        result = train_logistic_regression(X, y, epochs=epochs, lr=lr, seed=seed, backend=backend)
    elif model == "mlp":
        hidden = max(1, min(int(hidden), MAX_HIDDEN))
        result = train_mlp(X, y, hidden=hidden, epochs=epochs, lr=lr, seed=seed, backend=backend)
    else:
        raise ValueError(f"unknown model: {model!r} (expected 'logistic_regression' or 'mlp')")
    # Trim the loss history in the tool response to keep payloads bounded.
    history = result["loss_history"]
    if len(history) > 200:
        step = len(history) // 200
        result["loss_history"] = history[::step]
    return result
