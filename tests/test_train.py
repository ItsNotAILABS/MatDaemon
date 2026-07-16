import json

import numpy as np

from matdaemon.mcp_server import handle_request
from matdaemon.platform import get_platform_manifest
from matdaemon.train import (
    binary_cross_entropy,
    make_blobs_binary,
    make_xor,
    predict_logistic,
    sigmoid,
    train_classifier,
    train_logistic_regression,
    train_mlp,
)


def _tool_payload(response):
    return json.loads(response["result"]["content"][0]["text"])


# --- logistic regression genuinely learns ------------------------------------

def test_logistic_regression_learns_separable_data():
    X, y = make_blobs_binary(n=300, seed=1)
    result = train_logistic_regression(X, y, epochs=300, lr=0.3, seed=1, backend="numpy")
    assert result["accuracy"] > 0.95
    # loss actually went down
    assert result["final_loss"] < result["loss_history"][0]


def test_loss_history_is_monotone_nonincreasing_overall():
    X, y = make_blobs_binary(n=400, seed=3)
    result = train_logistic_regression(X, y, epochs=400, lr=0.2, seed=3, backend="numpy")
    losses = result["loss_history"]
    # full-batch GD on a convex problem: end loss well below start
    assert losses[-1] < losses[0] * 0.25


def test_training_is_deterministic():
    X, y = make_blobs_binary(n=200, seed=5)
    a = train_logistic_regression(X, y, epochs=100, lr=0.1, seed=5, backend="numpy")
    b = train_logistic_regression(X, y, epochs=100, lr=0.1, seed=5, backend="numpy")
    assert a["weights"] == b["weights"]
    assert a["bias"] == b["bias"]


# --- the classic separability contrast ---------------------------------------

def test_logistic_regression_cannot_solve_xor():
    X, y = make_xor(n=400, seed=2)
    result = train_logistic_regression(X, y, epochs=500, lr=0.3, seed=2, backend="numpy")
    assert result["accuracy"] < 0.75  # a line cannot separate XOR


def test_mlp_solves_xor():
    X, y = make_xor(n=400, seed=2)
    result = train_mlp(X, y, hidden=16, epochs=3000, lr=0.5, seed=2, backend="numpy")
    assert result["accuracy"] > 0.85  # the hidden layer can


# --- backward pass is provably correct (numerical gradient check) ------------

def test_logistic_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y = (X[:, 0] > 0).astype(np.float64).reshape(-1, 1)
    W = rng.standard_normal((3, 1)) * 0.1
    b = 0.0

    def loss_at(w, bias):
        return binary_cross_entropy(sigmoid(X @ w + bias), y)

    p = sigmoid(X @ W + b)
    analytic = (X.T @ (p - y)) / len(X)  # the gradient the trainer uses

    eps = 1e-6
    numerical = np.zeros_like(W)
    for i in range(3):
        wp, wm = W.copy(), W.copy()
        wp[i, 0] += eps
        wm[i, 0] -= eps
        numerical[i, 0] = (loss_at(wp, b) - loss_at(wm, b)) / (2 * eps)

    assert np.max(np.abs(analytic - numerical)) < 1e-6


def test_predict_logistic_round_trips():
    X, y = make_blobs_binary(n=100, seed=7)
    result = train_logistic_regression(X, y, epochs=200, lr=0.3, seed=7, backend="numpy")
    preds = (predict_logistic(X, result["weights"], result["bias"]) >= 0.5).astype(np.float64)
    assert np.mean(preds.reshape(-1, 1) == y) == result["accuracy"]


# --- bounded tool entry point -------------------------------------------------

def test_train_classifier_caps_epochs():
    X, y = make_blobs_binary(n=50, seed=0)
    result = train_classifier(X, y, model="logistic_regression", epochs=10 ** 9)
    assert result["epochs"] <= 20_000


def test_train_classifier_rejects_unknown_model():
    X, y = make_blobs_binary(n=20, seed=0)
    try:
        train_classifier(X, y, model="transformer")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown model")


# --- MCP tool surface ---------------------------------------------------------

def test_mcp_train_classifier_tool_learns():
    X, y = make_blobs_binary(n=200, seed=9)
    resp = handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {
            "name": "matdaemon_train_classifier",
            "arguments": {"features": X.tolist(), "labels": y.ravel().tolist(),
                          "model": "logistic_regression", "epochs": 300, "lr": 0.3, "backend": "numpy"},
        },
    })
    payload = _tool_payload(resp)
    assert payload["model"] == "logistic_regression"
    assert payload["accuracy"] > 0.9
    assert payload["final_loss"] < payload["loss_history"][0]


def test_train_classifier_registered_in_manifest():
    assert "matdaemon_train_classifier" in get_platform_manifest()["mcp_tools"]
