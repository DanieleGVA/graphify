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


class TestWheelShipsWhatTheCodeReads:
    """The component is only reusable if what it needs at runtime is inside the
    package. It was not: setuptools does not imply a subpackage from its parent,
    so the built wheel carried neither graphify_ent.server (the MCP server) nor
    graphify_ent.recipes, nor schema.cypher, nor ingredients.yaml. A wheel like
    that imports cleanly and fails on first real use.

    Verified against pyproject rather than the filesystem: the filesystem is
    always right in the checkout, which is exactly why the gap was invisible.
    """

    @staticmethod
    def _pyproject() -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()

    def test_every_subpackage_is_declared(self):
        src = self._pyproject()
        for pkg in ("graphify_ent.recipes", "graphify_ent.server"):
            assert f'"{pkg}"' in src, pkg

    def test_runtime_data_files_are_declared(self):
        src = self._pyproject()
        assert 'graphify_ent = ["schema.cypher"]' in src
        assert '"graphify_ent.recipes" = ["*.yaml"]' in src

    def test_the_declared_data_files_exist(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent / "graphify_ent"
        assert (root / "schema.cypher").exists()
        assert (root / "recipes" / "ingredients.yaml").exists()

    def test_the_registry_loads_from_the_package_not_the_cwd(self):
        """`Registry.load()` with no argument must find its YAML through the
        package, or it works in the checkout and breaks everywhere else."""
        from graphify_ent.recipes.ingredients import DEFAULT_REGISTRY
        assert DEFAULT_REGISTRY.name == "ingredients.yaml"
        assert DEFAULT_REGISTRY.parent.name == "recipes"


class TestComponentCommands:
    def test_the_cli_exposes_the_recipe_surface(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "graphify_ent" / "cli.py").read_text()
        for cmd in ('sub.add_parser("parse"', 'sub.add_parser("match"',
                    'sub.add_parser("query"', 'sub.add_parser("verify"',
                    'sub.add_parser("health"', 'sub.add_parser("mcp"'):
            assert cmd in src, cmd

    def test_parse_needs_no_database(self):
        """The cheapest integration for a project that has no Neo4j: reading an
        export with this component's vocabulary must not open a connection."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "graphify_ent" / "cli.py").read_text()
        body = src[src.index("def cmd_parse"):src.index("def cmd_match")]
        assert "_service()" not in body
        assert "Neo4jLoader" not in body
