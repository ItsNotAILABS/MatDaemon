import numpy as np

from matdaemon.moe import (
    MixtureOfExperts,
    make_regional_tasks,
    router_predict,
    softmax,
    train_softmax_router,
)
from matdaemon.train import train_logistic_regression, sigmoid
from matdaemon.matdaemon import matmul


def test_softmax_rows_sum_to_one():
    P = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.allclose(P[1], [1 / 3, 1 / 3, 1 / 3])  # uniform on ties


def test_router_learns_separable_clusters():
    X, y, dom = make_regional_tasks(n=600, seed=1)
    router = train_softmax_router(X, dom, num_experts=3, epochs=400, lr=0.3, seed=1, backend="numpy")
    routes = router_predict(router, X, backend="numpy")
    assert np.mean(routes == dom) > 0.95  # clusters are separable in context space
    assert router["loss_history"][-1] < router["loss_history"][0]


def test_moe_routes_correctly_and_is_accurate():
    X, y, dom = make_regional_tasks(n=1500, seed=0)
    Xtr, ytr, dtr = X[:1000], y[:1000], dom[:1000]
    Xte, yte, dte = X[1000:], y[1000:], dom[1000:]
    moe = MixtureOfExperts(expert="mlp", hidden=8, expert_epochs=1200, router_epochs=400,
                           lr=0.3, seed=0, backend="numpy").fit(Xtr, ytr, dtr)
    ev = moe.evaluate(Xte, yte, dte)
    assert ev["num_experts"] == 3
    assert ev["routing_accuracy"] > 0.95
    assert ev["accuracy"] > 0.9


def test_moe_beats_single_linear_model_on_conflicting_rules():
    # The whole point: specialists + router handle conflicting per-domain rules
    # that a single linear model cannot.
    X, y, dom = make_regional_tasks(n=1500, seed=2)
    Xtr, ytr, dtr = X[:1000], y[:1000], dom[:1000]
    Xte, yte = X[1000:], y[1000:]

    lr = train_logistic_regression(Xtr, ytr, epochs=800, lr=0.3, backend="numpy")
    W = np.asarray(lr["weights"]).reshape(-1, 1)
    single_acc = float(np.mean((sigmoid(matmul(Xte, W) + lr["bias"]).ravel() >= 0.5) == yte.ravel()))

    moe = MixtureOfExperts(expert="mlp", hidden=8, expert_epochs=1200, seed=2, backend="numpy").fit(Xtr, ytr, dtr)
    moe_acc = float(np.mean(moe.predict(Xte) == yte.ravel()))

    assert moe_acc > single_acc + 0.15  # a clear, comfortable margin


def test_moe_is_deterministic():
    X, y, dom = make_regional_tasks(n=600, seed=3)
    a = MixtureOfExperts(expert="logistic_regression", expert_epochs=200, seed=3, backend="numpy").fit(X, y, dom)
    b = MixtureOfExperts(expert="logistic_regression", expert_epochs=200, seed=3, backend="numpy").fit(X, y, dom)
    assert np.array_equal(a.predict(X), b.predict(X))


def test_make_regional_tasks_shapes():
    X, y, dom = make_regional_tasks(n=300, seed=0)
    assert X.shape[1] == 4
    assert set(dom.tolist()) == {0, 1, 2}
    assert y.shape == (300, 1)


def test_mcp_train_mixture_tool():
    import json

    from matdaemon.mcp_server import handle_request

    X, y, dom = make_regional_tasks(n=900, seed=0)
    resp = handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {
            "name": "matdaemon_train_mixture",
            "arguments": {
                "features": X.tolist(), "labels": y.ravel().tolist(), "domains": dom.ravel().tolist(),
                "expert": "mlp", "hidden": 8, "expert_epochs": 800, "backend": "numpy",
            },
        },
    })
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["num_experts"] == 3
    assert payload["routing_accuracy"] > 0.95
    assert payload["accuracy"] > 0.9


def test_train_mixture_registered_in_manifest():
    from matdaemon.platform import get_platform_manifest
    assert "matdaemon_train_mixture" in get_platform_manifest()["mcp_tools"]
