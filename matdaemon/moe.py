"""Mixture-of-Experts on the MatDaemon matmul backend.

The "many small specialists + one orchestrator" pattern, built native on
MatDaemon: several small expert models each trained on their own slice of the
problem, plus a learned router (a softmax gating model) that decides which
expert handles a given input. Every model here -- the router and every expert
-- trains and predicts through matmul (see train.py).

These experts are small neural nets / logistic heads, not generative language
models. But the architecture is the real thing: specialization + routing,
with experts you can add, retrain, or swap independently. A single flat model
must fit every sub-task in one weight set; the mixture routes each input to a
specialist that only has to be good at its own domain.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .matdaemon import matmul
from .train import sigmoid, train_logistic_regression, train_mlp

MAX_EXPERTS = 64


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_softmax_router(
    X, domain_labels, num_experts: int, epochs: int = 400, lr: float = 0.2, seed: int = 0, backend: str = "auto"
) -> dict:
    """Train a multiclass softmax gate: input -> which expert.

    Forward ``P = softmax(X @ W + b)``; gradient ``X.T @ (P - Y) / n`` -- both
    matmuls. Returns the weights and per-step cross-entropy loss.
    """
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    labels = np.asarray(domain_labels, dtype=np.int64).ravel()
    n = X.shape[0]
    Y = np.zeros((n, num_experts))
    Y[np.arange(n), labels] = 1.0  # one-hot targets

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, num_experts)) * 0.01
    b = np.zeros((1, num_experts))
    loss_history: list[float] = []
    for _ in range(epochs):
        P = softmax(matmul(X, W, backend=backend) + b)
        loss_history.append(float(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1.0)), axis=1))))
        grad = matmul(X.T, (P - Y), backend=backend) / n
        W -= lr * grad
        b -= lr * np.mean(P - Y, axis=0, keepdims=True)
    return {"W": W, "b": b, "loss_history": loss_history, "num_experts": num_experts, "backend": backend}


def router_predict(router: dict, X, backend: str = "auto") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    P = softmax(matmul(X, router["W"], backend=backend) + router["b"])
    return np.argmax(P, axis=1)


class MixtureOfExperts:
    """A router over a family of small binary-classifier experts.

    fit() trains the router on (input -> domain) and one expert per domain on
    just that domain's samples. predict() routes each input to its expert.
    """

    def __init__(self, expert: str = "mlp", hidden: int = 12, expert_epochs: int = 1500,
                 router_epochs: int = 400, lr: float = 0.3, seed: int = 0, backend: str = "auto"):
        if expert not in ("mlp", "logistic_regression"):
            raise ValueError("expert must be 'mlp' or 'logistic_regression'")
        self.expert = expert
        self.hidden = hidden
        self.expert_epochs = expert_epochs
        self.router_epochs = router_epochs
        self.lr = lr
        self.seed = seed
        self.backend = backend
        self.router: Optional[dict] = None
        self.experts: dict[int, dict] = {}
        self.num_experts = 0

    def fit(self, X, y, domains) -> "MixtureOfExperts":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        domains = np.asarray(domains, dtype=np.int64).ravel()
        expert_ids = sorted(set(domains.tolist()))
        self.num_experts = len(expert_ids)
        if self.num_experts > MAX_EXPERTS:
            raise ValueError(f"too many experts ({self.num_experts} > {MAX_EXPERTS})")

        self.router = train_softmax_router(
            X, domains, self.num_experts, epochs=self.router_epochs, lr=self.lr,
            seed=self.seed, backend=self.backend,
        )
        for eid in expert_ids:
            mask = domains == eid
            Xe, ye = X[mask], y[mask]
            if self.expert == "mlp":
                self.experts[eid] = train_mlp(Xe, ye, hidden=self.hidden, epochs=self.expert_epochs,
                                              lr=self.lr, seed=self.seed + eid, backend=self.backend)
            else:
                self.experts[eid] = train_logistic_regression(Xe, ye, epochs=self.expert_epochs,
                                                              lr=self.lr, seed=self.seed + eid, backend=self.backend)
        return self

    def _expert_predict(self, eid: int, X) -> np.ndarray:
        params = self.experts[eid]
        X = np.asarray(X, dtype=np.float64)
        if self.expert == "mlp":
            W1 = np.asarray(params["params"]["W1"]); b1 = np.asarray(params["params"]["b1"])
            W2 = np.asarray(params["params"]["W2"]); b2 = params["params"]["b2"]
            H = np.maximum(matmul(X, W1, backend=self.backend) + b1, 0.0)
            return sigmoid(matmul(H, W2, backend=self.backend) + b2).ravel()
        W = np.asarray(params["weights"]).reshape(-1, 1)
        return sigmoid(matmul(X, W, backend=self.backend) + params["bias"]).ravel()

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        routes = router_predict(self.router, X, backend=self.backend)
        out = np.zeros(X.shape[0])
        for eid in self.experts:
            mask = routes == eid
            if mask.any():
                out[mask] = (self._expert_predict(eid, X[mask]) >= 0.5).astype(np.float64)
        return out

    def evaluate(self, X, y, domains) -> dict:
        y = np.asarray(y, dtype=np.float64).ravel()
        domains = np.asarray(domains, dtype=np.int64).ravel()
        routes = router_predict(self.router, np.asarray(X, dtype=np.float64), backend=self.backend)
        preds = self.predict(X)
        return {
            "accuracy": float(np.mean(preds == y)),
            "routing_accuracy": float(np.mean(routes == domains)),
            "num_experts": self.num_experts,
            "expert": self.expert,
        }


# ---------------------------------------------------------------------------
# Deterministic multi-domain demo dataset
# ---------------------------------------------------------------------------


def make_regional_tasks(n: int = 900, seed: int = 0, noise: float = 0.25):
    """Three domains with conflicting per-domain rules on shared task features.

    Each sample carries context features (which cluster it's in -> its domain)
    and task features. The label rule differs by domain -- one is linear on
    t0, one linear on t1, one XOR of both -- so a single flat model must fit
    three conflicting rules, while a router + specialists each fit just one.
    Returns X = [c0, c1, t0, t1], binary y, and integer domain ids.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[-3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]])
    per = n // 3
    X_rows, y_rows, dom_rows = [], [], []
    for d in range(3):
        c = centres[d] + rng.normal(0, noise, size=(per, 2))
        t = rng.uniform(-1, 1, size=(per, 2))
        if d == 0:
            label = (t[:, 0] > 0)
        elif d == 1:
            label = (t[:, 1] > 0)
        else:
            label = ((t[:, 0] > 0) ^ (t[:, 1] > 0))
        X_rows.append(np.hstack([c, t]))
        y_rows.append(label.astype(np.float64))
        dom_rows.append(np.full(per, d))
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows).reshape(-1, 1)
    dom = np.concatenate(dom_rows)
    perm = rng.permutation(len(X))
    return X[perm], y[perm], dom[perm]
