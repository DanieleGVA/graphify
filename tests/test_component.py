"""The component doorway: CLI + MCP over the existing service, no database.

What these tests pin, and why:

  * `--help` and usage errors must work with no Neo4j reachable — a component
    that cannot even print its usage without infrastructure is not a component.
  * The MCP factory registers exactly the advertised tools; a rename here is an
    API break for every consuming project, so the list is asserted verbatim.
  * `verify_card` produces the same report shape the T73/T74 harness writes —
    one format, wherever the verdict comes from.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from graphify_ent.cli import main, verify_card

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class _Hit:
    def __init__(self, i):
        self.node_id = str(i)


class _Retriever:
    """Fixed passages, as in test_verify — the plumbing is what is under test."""

    def __init__(self, docs):
        self.docs = docs

    def query(self, *a, **kw):
        r = type("R", (), {})()
        r.refused = not self.docs
        r.hits = [_Hit(i) for i in range(len(self.docs))]
        r.channel_counts = {"fast_path": 1}
        return r

    def hydrate(self, ids):
        return {str(i): d for i, d in enumerate(self.docs)}


class _Service:
    def __init__(self, docs=()):
        self.retriever = _Retriever(list(docs))

    def _embed(self, text):
        return None

    def health(self):
        return {"ok": True}


BECHAMEL = {"passage": "Béchamel Sauce White Roux 1 lb 454 g Milk 5 qt 4.80 L",
            "source_file": "book.pdf", "source_location": "pages 3-3",
            "text_excerpt": "Béchamel"}


class TestCliWithoutInfrastructure:
    def test_help_needs_no_database(self, capsys):
        with pytest.raises(SystemExit) as e:
            main(["--help"])
        assert e.value.code == 0
        assert "graphify-ent" in capsys.readouterr().out

    def test_missing_subcommand_is_usage_error(self):
        with pytest.raises(SystemExit) as e:
            main([])
        assert e.value.code == 2

    def test_health_exit_code_follows_ok(self, monkeypatch, capsys):
        monkeypatch.setattr("graphify_ent.cli._service", lambda: _Service())
        assert main(["health"]) == 0
        assert json.loads(capsys.readouterr().out)["ok"] is True

    def test_operational_failure_is_exit_1_not_traceback(self, monkeypatch, capsys):
        monkeypatch.setattr("graphify_ent.cli._service",
                            lambda: (_ for _ in ()).throw(ConnectionError("no db")))
        assert main(["query", "anything"]) == 1
        assert "ConnectionError" in capsys.readouterr().err


class TestVerifyCard:
    def test_report_shape_matches_the_harness(self):
        card = {"title": "t", "claimed_reference": "book",
                "claims": [{"subject": "Bechamel Sauce", "aspect": "white roux",
                            "value": "454 g"}]}
        report = verify_card(_Service([BECHAMEL]), card)
        assert report["counts"]["SUPPORTED"] == 1
        assert report["used_pdf"] is False
        f = report["findings"][0]
        assert f["verdict"] == "SUPPORTED" and f["source_file"] == "book.pdf"

    def test_empty_corpus_confirms_nothing(self):
        card = {"claims": [{"subject": "Bechamel Sauce", "aspect": "white roux"}]}
        report = verify_card(_Service([]), card)
        assert report["counts"]["SUPPORTED"] == 0


class TestMcpFactory:
    def test_registers_the_advertised_tools(self):
        from graphify_ent.server.mcp import build_mcp
        mcp = build_mcp(service=_Service())
        tools = {t.name for t in asyncio.run(mcp.list_tools())}
        assert tools == {"query_graph", "get_node", "get_neighbors",
                         "shortest_path", "list_domains", "verify_claim",
                         "health"}

    def test_verify_claim_tool_answers_from_the_stub(self):
        from graphify_ent.server.mcp import build_mcp
        mcp = build_mcp(service=_Service([BECHAMEL]))
        out = asyncio.run(mcp.call_tool(
            "verify_claim", {"subject": "Bechamel Sauce",
                             "aspect": "white roux", "value": "454 g"}))
        payload = json.loads(out[0][0].text) if isinstance(out, tuple) else \
            json.loads(out[0].text)
        assert payload["verdict"] == "SUPPORTED"
