import time

from fastapi.testclient import TestClient

from matdaemon.api import create_app


def test_api_health():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_platform_manifest():
    client = TestClient(create_app())
    response = client.get("/v1/platform")
    assert response.status_code == 200
    payload = response.json()
    surface_ids = {surface["id"] for surface in payload["surfaces"]}
    assert payload["name"] == "MatDaemon"
    assert "api" in surface_ids
    assert "mcp" in surface_ids
    assert payload["operator_commands"]["serve_api"].startswith("matdaemon serve")


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


def test_api_use_cases():
    client = TestClient(create_app())
    response = client.get("/v1/use-cases")
    assert response.status_code == 200
    payload = response.json()
    assert any(case["id"] == "agent-memory-routing" for case in payload["use_cases"])


def test_api_async_job_lifecycle():
    client = TestClient(create_app())
    response = client.post(
        "/v1/jobs/matmul",
        json={
            "a": [[1, 2], [3, 4]],
            "b": [[5, 6], [7, 8]],
            "backend": "numpy",
            "use_case": "agent-memory-routing",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = None
    for _ in range(30):
        status_response = client.get(f"/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        if status["status"] == "completed":
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "completed"
    assert status["shape"] == [2, 2]
    assert status["use_case"] == "agent-memory-routing"

    result_response = client.get(f"/v1/jobs/{job_id}/result")
    assert result_response.status_code == 200
    assert result_response.json()["result"] == [[19.0, 22.0], [43.0, 50.0]]
