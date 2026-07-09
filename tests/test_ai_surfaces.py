import json

from matdaemon.mcp_server import create_mcp_server, handle_request
from matdaemon.platform import get_platform_manifest
from matdaemon.use_cases import USE_CASES, get_use_case


def _tool_payload(response):
    return json.loads(response["result"]["content"][0]["text"])


def test_use_case_registry_contains_ai_routes():
    ids = {case["id"] for case in USE_CASES}
    assert "agent-memory-routing" in ids
    assert "local-rag-similarity" in ids
    assert get_use_case("attention-block")["recommended_backend"] == "tiled"


def test_platform_manifest_contains_ai_surfaces():
    manifest = get_platform_manifest()
    surface_ids = {surface["id"] for surface in manifest["surfaces"]}
    assert "mcp" in surface_ids
    assert "github-action" in surface_ids
    assert "matdaemon_backend_status" in manifest["mcp_tools"]
    assert any(case["id"] == "local-rag-similarity" for case in manifest["use_cases"])


def test_mcp_server_contract_is_self_contained():
    server = create_mcp_server()
    tool_names = {tool["name"] for tool in server["tools"]}
    assert server["serverInfo"]["name"] == "MatDaemon"
    assert "matdaemon_matmul" in tool_names
    assert "matdaemon_backend_status" in tool_names
    assert "matdaemon_smoke_benchmark" in tool_names


def test_mcp_initialize_and_list_tools():
    initialize = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert initialize["result"]["serverInfo"]["name"] == "MatDaemon"

    listed = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tool_names = {tool["name"] for tool in listed["result"]["tools"]}
    assert "matdaemon_generate_api_payload" in tool_names


def test_mcp_matmul_tool_call():
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "matdaemon_matmul",
                "arguments": {"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]], "backend": "numpy"},
            },
        }
    )
    payload = _tool_payload(response)
    assert payload["result"] == [[19.0, 22.0], [43.0, 50.0]]
    assert payload["shape"] == [2, 2]


def test_mcp_payload_generation_tool_call():
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "matdaemon_generate_api_payload", "arguments": {"backend": "numpy"}},
        }
    )
    payload = _tool_payload(response)
    assert payload["endpoint"] == "/v1/jobs/matmul"
    assert payload["body"]["backend"] == "numpy"
