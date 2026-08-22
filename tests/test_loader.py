"""Phase 2 — Neo4j schema + bulk loader.

Acceptance (execution plan §2): pilot graph loads in seconds; re-running the
loader is idempotent (counts unchanged); >10k nodes/s and >20k edges/s vs the
autocommit baseline.

Integration tests need a live Neo4j. They are skipped unless NEO4J_URI is set
(the harness in `conftest_neo4j.py` starts one via Docker when available), so
the suite stays green on a machine without Docker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from graphify_ent.loader import (
    Neo4jLoader,
    iter_edges,
    iter_nodes,
    safe_label,
    safe_rel,
    sanitize_props,
)

NEO4J_URI = os.environ.get("NEO4J_URI")
requires_neo4j = pytest.mark.skipif(not NEO4J_URI, reason="NEO4J_URI not set")


# ---------------------------------------------------------------- unit tests

class TestSanitization:
    """Upstream's _safe_rel/_safe_label discipline is kept verbatim (plan §2)."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("calls", "CALLS"),
            ("conceptually related to", "CONCEPTUALLY_RELATED_TO"),
            ("shares-data-with", "SHARES_DATA_WITH"),
            ("DROP DATABASE", "DROP_DATABASE"),
            ("", "RELATED_TO"),
            ("`) DETACH DELETE n //", "___DETACH_DELETE_N___"),
        ],
    )
    def test_safe_rel(self, raw, expected):
        assert safe_rel(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [("Document", "Document"), ("", "Entity"), ("Foo:Bar", "FooBar"), ("a`b", "ab")],
    )
    def test_safe_label(self, raw, expected):
        assert safe_label(raw) == expected

    def test_props_keep_only_primitives(self):
        props = sanitize_props({"a": 1, "b": "x", "c": [1, 2], "d": {"k": 1}, "e": None, "f": 1.5})
        assert props == {"a": 1, "b": "x", "f": 1.5}

    def test_props_never_carry_injection_via_key(self):
        props = sanitize_props({"ok": 1, "bad key`": 2, "with space": 3})
        assert "ok" in props and "bad key`" not in props and "with space" not in props


class TestInputParsing:
    def test_reads_graph_json(self, tmp_path):
        p = tmp_path / "graph.json"
        p.write_text(json.dumps({
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b", "relation": "cites"}],
        }))
        assert len(list(iter_nodes(p))) == 2
        assert len(list(iter_edges(p))) == 1

    def test_reads_jsonl_staging_format(self, tmp_path):
        nodes = tmp_path / "nodes.jsonl"
        edges = tmp_path / "edges.jsonl"
        nodes.write_text('{"id":"a","label":"A"}\n{"id":"b","label":"B"}\n')
        edges.write_text('{"source":"a","target":"b","relation":"cites"}\n')
        assert len(list(iter_nodes(nodes))) == 2
        assert len(list(iter_edges(edges))) == 1

    def test_blank_lines_are_skipped(self, tmp_path):
        p = tmp_path / "nodes.jsonl"
        p.write_text('{"id":"a"}\n\n   \n{"id":"b"}\n')
        assert len(list(iter_nodes(p))) == 2

    def test_networkx_node_link_shape(self, tmp_path):
        p = tmp_path / "graph.json"
        p.write_text(json.dumps({
            "nodes": [{"id": "a"}],
            "links": [{"source": "a", "target": "a", "relation": "self"}],
        }))
        assert len(list(iter_edges(p))) == 1


class TestBatching:
    def test_batches_respect_size(self):
        from graphify_ent.loader import batched

        out = list(batched(range(10), 3))
        assert [len(b) for b in out] == [3, 3, 3, 1]

    def test_empty_input_yields_nothing(self):
        from graphify_ent.loader import batched

        assert list(batched([], 5)) == []


# --------------------------------------------------------- integration tests

@requires_neo4j
class TestLoadAgainstNeo4j:
    @pytest.fixture
    def loader(self):
        ldr = Neo4jLoader(
            uri=NEO4J_URI,
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "enterpriphy"),
        )
        ldr.wipe()
        ldr.apply_schema()
        yield ldr
        ldr.close()

    @pytest.fixture
    def graph_file(self, tmp_path) -> Path:
        nodes = [{"id": f"n{i}", "label": f"Node {i}", "file_type": "document",
                  "source_file": f"doc{i % 3}.pdf", "text_excerpt": f"excerpt {i}"}
                 for i in range(500)]
        edges = [{"source": f"n{i}", "target": f"n{(i + 1) % 500}", "relation": "references",
                  "confidence": "EXTRACTED", "weight": 1.0} for i in range(500)]
        p = tmp_path / "graph.json"
        p.write_text(json.dumps({"nodes": nodes, "edges": edges}))
        return p

    def test_schema_is_idempotent(self, loader):
        loader.apply_schema()
        loader.apply_schema()  # must not raise
        assert "entity_id" in loader.list_constraints()

    def test_constraints_exist_before_load(self, loader, graph_file):
        loader.load(graph_file, domain="test")
        assert "entity_id" in loader.list_constraints()

    def test_loads_all_nodes_and_edges(self, loader, graph_file):
        stats = loader.load(graph_file, domain="test")
        assert stats.nodes_written == 500
        assert stats.edges_written == 500
        counts = loader.counts()
        assert counts["nodes"] == 500
        assert counts["edges"] == 500

    def test_reconciliation_reports_match(self, loader, graph_file):
        stats = loader.load(graph_file, domain="test")
        assert stats.reconciled is True
        assert stats.unmatched_edges == 0

    def test_edge_with_missing_endpoint_is_reported_not_silently_lost(self, loader, tmp_path):
        """The silent data-loss path the reconciliation gate exists to catch."""
        p = tmp_path / "g.json"
        p.write_text(json.dumps({
            "nodes": [{"id": "a", "label": "A"}],
            "edges": [{"source": "a", "target": "ghost", "relation": "cites"}],
        }))
        stats = loader.load(p, domain="test")
        assert stats.edges_written == 0, "an unmatched edge must not count as written"
        assert stats.unmatched_edges == 1
        assert stats.reconciled is False

    def test_duplicate_input_rows_still_reconcile(self, loader, tmp_path):
        """MERGE collapses duplicates; that is not a reconciliation failure."""
        p = tmp_path / "dup.json"
        edge = {"source": "a", "target": "b", "relation": "cites"}
        p.write_text(json.dumps({
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "a"}],
            "edges": [edge, dict(edge)],
        }))
        stats = loader.load(p, domain="test")
        assert stats.reconciled is True
        assert loader.counts()["edges"] == 1

    def test_rerun_is_idempotent(self, loader, graph_file):
        loader.load(graph_file, domain="test")
        first = loader.counts()
        loader.load(graph_file, domain="test")
        assert loader.counts() == first, "re-running must not duplicate"

    def test_domain_is_stamped_on_every_node(self, loader, graph_file):
        loader.load(graph_file, domain="pilot")
        assert loader.count_where("n.domain = 'pilot'") == 500

    def test_dry_run_writes_nothing(self, loader, graph_file):
        stats = loader.load(graph_file, domain="test", dry_run=True)
        assert stats.nodes_written == 500  # planned
        assert loader.counts()["nodes"] == 0  # but nothing persisted

    def test_ingested_at_is_stamped(self, loader, graph_file):
        loader.load(graph_file, domain="test")
        assert loader.count_where("n.ingested_at IS NOT NULL") == 500

    def test_throughput_meets_acceptance(self, loader, tmp_path):
        """>10k nodes/s and >20k edges/s (execution plan §2)."""
        n = 20_000
        nodes = [{"id": f"b{i}", "label": f"B{i}", "file_type": "document"} for i in range(n)]
        edges = [{"source": f"b{i}", "target": f"b{(i + 1) % n}", "relation": "references"}
                 for i in range(2 * n // 1)]
        p = tmp_path / "bench.json"
        p.write_text(json.dumps({"nodes": nodes, "edges": edges}))

        stats = loader.load(p, domain="bench")
        assert stats.nodes_per_s > 10_000, f"only {stats.nodes_per_s:.0f} nodes/s"
        assert stats.edges_per_s > 20_000, f"only {stats.edges_per_s:.0f} edges/s"
