import json

import matdaemon as md
from matdaemon.cli import main
from matdaemon.platform import get_platform_manifest


def test_platform_manifest_contains_product_surfaces():
    manifest = get_platform_manifest()
    surface_ids = {surface["id"] for surface in manifest["surfaces"]}

    assert manifest["name"] == "MatDaemon"
    assert manifest["version"] == "0.3.0"
    assert manifest["status"] == "production-beta"
    assert {"sdk", "api", "mcp", "github-action", "cuda-backend"}.issubset(surface_ids)
    assert any(gate["gate"] == "benchmark" for gate in manifest["proof_gates"])


def test_sdk_exports_platform_manifest():
    manifest = md.get_platform_manifest()
    assert manifest["operator_commands"]["show_manifest"] == "matdaemon platform"
    assert any(surface["id"] == "daemon" for surface in md.PLATFORM_SURFACES)


def test_cli_platform_command_prints_manifest(capsys):
    exit_code = main(["platform"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["name"] == "MatDaemon"
    assert any(surface["id"] == "mcp" for surface in payload["surfaces"])
