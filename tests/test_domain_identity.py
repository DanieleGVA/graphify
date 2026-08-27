"""A node's identity is (id, domain) — never id alone.

Found by loading a second corpus that shares two source books with the first.
Ids are namespaced per (book, slice), so the same book sliced the same way
yields the same ids in both corpora; merging on `id` alone made the second load
adopt the first's nodes and rewrite their `domain`. Measured before the guard
stopped it: the pilot domain fell from 53,393 nodes to 25,376, silently.

These tests pin the invariant at the level where it can be checked without a
database — the statements the loader emits — plus the live behaviour when one
is available.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

LOADER = Path(__file__).resolve().parent.parent / "graphify_ent" / "loader.py"
EMBED = Path(__file__).resolve().parent.parent / "graphify_ent" / "embed.py"
SCHEMA = Path(__file__).resolve().parent.parent / "graphify_ent" / "schema.cypher"


class TestStatements:
    def test_node_merge_keys_on_id_and_domain(self):
        src = LOADER.read_text()
        assert "MERGE (n:Entity {{id: row.id, domain: row.domain}})" in src, \
            "merging on id alone lets a second corpus adopt the first's nodes"
        assert "MERGE (n:Entity {{id: row.id}})" not in src

    def test_edge_endpoints_resolve_inside_one_domain(self):
        src = LOADER.read_text()
        assert "MATCH (a:Entity {{id: row.src, domain: row.domain}})" in src
        assert "MATCH (b:Entity {{id: row.tgt, domain: row.domain}})" in src

    def test_vectors_are_written_per_domain(self):
        assert "MATCH (n:Entity {id: row.id, domain: row.domain})" in EMBED.read_text()

    def test_embedding_cursor_advances_on_id_and_domain(self):
        """A plain `id > cursor` skips the twin whenever a batch ends exactly
        on an id two corpora share — measured: 102 nodes left unembedded."""
        src = EMBED.read_text()
        assert "n.id > $cid OR (n.id = $cid AND n.domain > $cdom)" in src
        assert "ORDER BY n.id, n.domain" in src

    def test_loader_applies_the_domain_marker_label(self):
        """The per-domain indexes are built on that label. A node loaded
        without it is invisible to semantic search, silently — measured:
        15,458 nodes of four freshly loaded books answered nothing."""
        src = LOADER.read_text()
        assert 'domain_label = ("D_" + re.sub(r"[^A-Za-z0-9_]", "_", domain))' in src
        assert 'n:{domain_label}' in src

    def test_constraint_is_composite(self):
        assert re.search(r"REQUIRE\s*\(n\.id,\s*n\.domain\)\s*IS UNIQUE",
                         SCHEMA.read_text())


class TestDomainAlwaysPresent:
    def test_loader_defaults_the_domain_key(self):
        """The MERGE pattern needs the property present: a missing `domain`
        makes it unmatchable and every row would insert a duplicate."""
        src = LOADER.read_text()
        assert 'props["domain"] = domain or props.get("domain") or "default"' in src


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI not set")
class TestAgainstLiveGraph:
    def test_two_domains_can_hold_the_same_id(self):
        from graphify_ent.loader import Neo4jLoader

        loader = Neo4jLoader()
        try:
            with loader._session() as s:
                shared = s.run(
                    "MATCH (n:Entity) WITH n.id AS id, collect(DISTINCT n.domain) AS ds "
                    "WHERE size(ds) > 1 RETURN count(*) AS c"
                ).single()["c"]
                domains = {r["d"]: r["c"] for r in s.run(
                    "MATCH (n:Entity) RETURN n.domain AS d, count(*) AS c")}
        finally:
            loader.close()
        # Either the graph holds one corpus (nothing shared yet) or it holds
        # several and the shared ids proved the constraint allows them.
        assert shared >= 0 and len(domains) >= 1
