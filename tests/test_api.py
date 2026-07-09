from fastapi.testclient import TestClient

from matdaemon.api import create_app


def test_api_health():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_matmul():
    client = TestClient(create_app())
    response = client.post(
        "/v1/matmul",
        json={
            "a": [[1, 2], [3, 4]],
            "b": [[5, 6], [7, 8]],
            "backend": "numpy",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["shape"] == [2, 2]
    assert payload["result"] == [[19.0, 22.0], [43.0, 50.0]]
