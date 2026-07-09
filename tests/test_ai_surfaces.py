from matdaemon.platform import get_platform_manifest
from matdaemon.use_cases import USE_CASES, get_use_case


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
    assert any(case["id"] == "local-rag-similarity" for case in manifest["use_cases"])


def test_mcp_server_module_imports_without_mcp_extra():
    import matdaemon.mcp_server as mcp_server

    assert hasattr(mcp_server, "create_mcp_server")
    assert hasattr(mcp_server, "main")
