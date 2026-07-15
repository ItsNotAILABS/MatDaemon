import json
import subprocess
import sys

import numpy as np

from matdaemon.mcp_server import handle_request
from matdaemon.platform import get_platform_manifest
from matdaemon.text import hashing_embed, text_similarity_top_k


def _tool_payload(response):
    return json.loads(response["result"]["content"][0]["text"])


def test_hashing_embed_is_deterministic_in_process():
    a = hashing_embed(["quarterly financial report"], dim=64)
    b = hashing_embed(["quarterly financial report"], dim=64)
    assert np.array_equal(a, b)


def test_hashing_embed_is_deterministic_across_processes():
    # The whole point of using blake2b instead of builtin hash(): identical
    # output even under a different PYTHONHASHSEED in a separate interpreter.
    code = (
        "import numpy as np;from matdaemon.text import hashing_embed;"
        "print(hashing_embed(['q3 report'], dim=32).round(6).tolist())"
    )
    out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"})
    out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert out1.returncode == 0, out1.stderr
    assert out1.stdout == out2.stdout


def test_hashing_embed_shape_and_unit_norm():
    v = hashing_embed(["one two three", "four"], dim=32)
    assert v.shape == (2, 32)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_identical_text_scores_one():
    res = text_similarity_top_k(["Q3 report"], ["Q3 report", "unrelated matter"], k=1)
    assert res["top_k"][0][0] == 0
    assert res["top_scores"][0][0] > 0.99


def test_shared_words_rank_above_unrelated():
    res = text_similarity_top_k(
        ["quarterly financial report"],
        ["annual financial report", "website redesign launch"],
        k=2,
    )
    assert res["top_k"][0][0] == 0  # the one sharing "financial report"
    assert res["top_scores"][0][0] > res["top_scores"][0][1]


def test_char_ngram_catches_morphology():
    # "launching" and "launch" share no whole word but share character 5-grams.
    word_only = text_similarity_top_k(["launching the site"], ["launch"], k=1, char_ngram=0)
    with_char = text_similarity_top_k(["launching the site"], ["launch"], k=1, char_ngram=5)
    assert with_char["top_scores"][0][0] > word_only["top_scores"][0][0]


def test_empty_candidates_raises():
    try:
        text_similarity_top_k(["x"], [], k=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty candidates")


def test_mcp_embed_text_tool():
    resp = handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "matdaemon_embed_text", "arguments": {"texts": ["a b", "c"], "dim": 16}},
    })
    payload = _tool_payload(resp)
    assert payload["shape"] == [2, 16]
    assert payload["deterministic"] is True


def test_mcp_text_similarity_tool():
    resp = handle_request({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {
            "name": "matdaemon_text_similarity_top_k",
            "arguments": {"queries": ["hello world"], "candidates": ["hello world", "xyz abc"], "k": 1},
        },
    })
    payload = _tool_payload(resp)
    assert payload["top_k"][0][0] == 0
    assert payload["top_scores"][0][0] > 0.99


def test_new_tools_registered_in_manifest():
    tools = get_platform_manifest()["mcp_tools"]
    assert "matdaemon_embed_text" in tools
    assert "matdaemon_text_similarity_top_k" in tools
