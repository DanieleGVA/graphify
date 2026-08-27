"""The evidence lane: the page layer gets its own candidate slots.

Measured on the sixteen-book corpus before this existed (`evidence/T95`,
`docs/diagnosi-accuratezza-16-libri.md`): grounding stopped at 84.0%, and of
the 32 failures **28 were pages that never entered the candidate set** — 90 to
100% of it was concept nodes, which outnumber pages ten to one and are ranked
against them in the same index. Restricting retrieval to pages alone recovered
12 of 16 sampled failures, which is the counter-proof that the answer was in
the graph the whole time and only the shortlist was wrong.

`balance_by_method` shares out the *window*; by then it is too late, because a
page absent from the candidates cannot be balanced into anything. So the share
moves to where candidates are generated: one index per lane, four ranked lists,
one RRF.

These tests pin the properties that make the lane safe to ship:
  * it is ablatable (`ENTERPRIPHY_EVIDENCE_LANE=0`) — the plan requires the
    ablation to be able to overrule the design;
  * a graph without page indexes silently loses the lane and NEVER falls back
    to the domain-wide index, which would make the lane a copy of the thing it
    complements;
  * the lanes never mix corpora, the failure the shared index taught.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

from graphify_ent.retrieval import (
    PAGE_LANE,
    HybridRetriever,
    evidence_lane_enabled,
)

ROOT = Path(__file__).resolve().parent.parent
LOADER = ROOT / "graphify_ent" / "loader.py"
BUILD_INDEXES = ROOT / "tools" / "build_domain_indexes.py"
ENRICH = ROOT / "tools" / "enrich_passages.py"

NEO4J_URI = os.environ.get("NEO4J_URI")
requires_neo4j = pytest.mark.skipif(not NEO4J_URI, reason="NEO4J_URI not set")

BOTH_LANES = {
    "entity_embedding_pilot", "entity_text_pilot",
    "entity_embedding_pilot_pages", "entity_text_pilot_pages",
    "entity_embedding_canon_library", "entity_text_canon_library",
    "entity_embedding_canon_library_pages", "entity_text_canon_library_pages",
}
CONCEPT_ONLY = {n for n in BOTH_LANES if not n.endswith(PAGE_LANE)}


@pytest.fixture
def retriever(monkeypatch):
    """A retriever whose index roster is fixed, with no database behind it.

    `_domain_indexes` is a class attribute — a cache shared by every instance —
    so it is patched rather than assigned, or one test's roster leaks into the
    next.
    """
    def _make(names):
        monkeypatch.setattr(HybridRetriever, "_domain_indexes", set(names))
        return HybridRetriever(loader=None)
    return _make


class TestLaneSelection:
    def test_both_lanes_when_the_page_index_exists(self, retriever):
        assert retriever(BOTH_LANES).lanes("pilot") == ("", PAGE_LANE)

    def test_only_the_domain_lane_when_it_does_not(self, retriever):
        assert retriever(CONCEPT_ONLY).lanes("pilot") == ("",)

    def test_env_flag_ablates_the_lane(self, retriever, monkeypatch):
        """The plan's rule: if a re-measurement gets worse, the ablation
        decides, not the affection for the code."""
        r = retriever(BOTH_LANES)
        monkeypatch.setenv("ENTERPRIPHY_EVIDENCE_LANE", "0")
        assert evidence_lane_enabled() is False
        assert r.lanes("pilot") == ("",)

    def test_lane_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv("ENTERPRIPHY_EVIDENCE_LANE", raising=False)
        assert evidence_lane_enabled() is True

    def test_domainless_query_keeps_the_lane(self, retriever):
        """No shared index exists — it duplicated every vector and is what ran
        the database out of memory — so a domain-less query fans out over the
        per-domain ones, in both lanes."""
        assert retriever(BOTH_LANES).lanes(None) == ("", PAGE_LANE)


class TestIndexResolution:
    def test_page_index_names_follow_the_domain_ones(self, retriever):
        r = retriever(BOTH_LANES)
        assert r._domain_index("pilot", PAGE_LANE) == "entity_embedding_pilot_pages"
        assert r._text_index("pilot", PAGE_LANE) == "entity_text_pilot_pages"

    def test_missing_page_index_never_falls_back(self, retriever):
        """Falling back to the domain-wide index would make the page lane a
        second copy of the concept lane: the same candidates, ranked twice,
        and the failure this whole change exists to fix left in place."""
        r = retriever(CONCEPT_ONLY)
        assert r._domain_index("pilot", PAGE_LANE) == ""
        assert r._text_index("pilot", PAGE_LANE) == ""

    def test_domain_lane_still_falls_back_to_the_shared_index(self, retriever):
        """A graph that never ran build_domain_indexes keeps working."""
        r = retriever(set())
        assert r._domain_index("pilot") == "entity_embedding"
        assert r._text_index("pilot") == "entity_text"

    def test_absent_index_returns_no_candidates_without_querying(self, retriever):
        """`loader=None` is the assertion: touching the database would raise."""
        r = retriever(CONCEPT_ONLY)
        assert r.vector_search([0.1] * 8, domain="pilot", lane=PAGE_LANE) == []
        assert r.fulltext_search("roux", domain="pilot", lane=PAGE_LANE) == []
        assert r._lexical_evidence("roux", "pilot", 10, lane=PAGE_LANE) == ([], 0.0)

    def test_merged_indexes_keep_the_lanes_apart(self, retriever):
        r = retriever(BOTH_LANES)
        concepts = r._merged_vector_indexes()
        pages = r._merged_vector_indexes(PAGE_LANE)
        assert concepts == ["entity_embedding_canon_library", "entity_embedding_pilot"]
        assert pages == ["entity_embedding_canon_library_pages",
                         "entity_embedding_pilot_pages"]
        assert not set(concepts) & set(pages), "a node counted twice is ranked twice"


class TestPageLabelIsWrittenByTheLoader:
    """Mirror of `test_domain_identity`: an index is built on a label, so a
    node written without the label is invisible to the lane built for it —
    silently. That already happened once with the domain label: 15,458 nodes
    of four freshly loaded books answered nothing at all."""

    def test_loader_labels_page_nodes(self):
        src = LOADER.read_text()
        assert 'is_page = props.get("extraction_method") == PAGE_METHOD' in src
        assert 'labels += f", n:{PAGE_LABEL}"' in src

    def test_page_label_is_per_domain_too(self):
        assert 'labels += f", n:{domain_label}_pages"' in LOADER.read_text()

    def test_builder_creates_both_page_indexes(self):
        src = BUILD_INDEXES.read_text()
        assert "def page_index_for(domain: str) -> str:" in src
        assert 'return index_for(domain) + "_pages"' in src
        assert "CREATE VECTOR INDEX {pidx}" in src
        assert "_pages " in src and "CREATE FULLTEXT INDEX entity_text_" in src

    def test_builder_backfills_the_label_from_the_extraction_method(self):
        src = BUILD_INDEXES.read_text()
        assert "n.extraction_method = 'page'" in src
        assert "SET n:Page, n:{plbl}" in src


class TestSharedVectorIndexStaysDead:
    """It was dropped for a measured reason — a second copy of every vector,
    and the memory that killed the container during an HNSW build (exit 137) —
    and it came back anyway, because `schema.cypher` recreates it on every
    load. Found in the graph at 199,655 vectors, 100% populated, queried by
    nothing."""

    def test_apply_schema_skips_it_when_per_domain_indexes_exist(self):
        src = LOADER.read_text()
        assert "_SHARED_VECTOR_INDEX.search(stmt)" in src
        assert "if per_domain and" in src

    def test_the_pattern_matches_the_schema_statement(self):
        from graphify_ent.loader import _SHARED_VECTOR_INDEX
        schema = (ROOT / "graphify_ent" / "schema.cypher").read_text()
        stmt = next(s for s in schema.split(";") if "VECTOR INDEX" in s)
        assert _SHARED_VECTOR_INDEX.search(stmt)

    def test_the_pattern_does_not_match_a_per_domain_statement(self):
        from graphify_ent.loader import _SHARED_VECTOR_INDEX
        assert not _SHARED_VECTOR_INDEX.search(
            "CREATE VECTOR INDEX entity_embedding_pilot_pages IF NOT EXISTS "
            "FOR (n:D_pilot_pages) ON (n.embedding)")


class TestOverlappingPassages:
    """Four of the 32 grounding failures were quotes straddling a page break:
    no single page node contains them, so no retrieval could return one."""

    @pytest.fixture(autouse=True)
    def _tool(self):
        sys.path.insert(0, str(ROOT / "tools"))
        try:
            yield
        finally:
            sys.path.remove(str(ROOT / "tools"))

    def _fn(self):
        from enrich_passages import OVERLAP_MARK, overlap_passage
        return overlap_passage, OVERLAP_MARK

    def test_next_page_head_is_appended_and_marked(self):
        overlap, mark = self._fn()
        out = overlap("...cook the sugar to", "118 °C, then fold it in.", 400, 42)
        assert out.endswith("118 °C, then fold it in.")
        assert mark.format(page=42) in out, "borrowed text must announce itself"

    def test_running_it_twice_writes_the_same_value(self):
        """Idempotence by construction: the passage is rebuilt from the source
        each run, never appended to what is already stored."""
        overlap, _ = self._fn()
        once = overlap("page text", "next page", 400, 7)
        twice = overlap("page text", "next page", 400, 7)
        assert once == twice
        assert once.count("next page") == 1

    def test_width_zero_is_the_identity(self):
        overlap, mark = self._fn()
        assert overlap("page text", "next page", 0, 7) == "page text"

    def test_last_page_has_nothing_to_borrow(self):
        overlap, mark = self._fn()
        assert overlap("page text", "", 400, 7) == "page text"

    def test_borrowed_text_is_truncated_to_the_asked_width(self):
        overlap, mark = self._fn()
        out = overlap("p", "x" * 900, 400, 2)
        assert out.count("x") == 400

    def test_page_bounds_are_never_rewritten(self):
        """Provenance still names ONE page; the overlap is text, not a claim
        about where the text lives."""
        src = ENRICH.read_text()
        start = src.index("nxt = texts[hi]")
        block = src[start: src.index('if (r["m"] or "") == "native":', start)]
        assert '"lo": pg[0], "hi": pg[1]' in block
        assert "page_lo" not in block and "page_hi" not in block


@requires_neo4j
class TestAgainstLiveGraph:
    @pytest.fixture(scope="class")
    def live(self):
        from graphify_ent.loader import Neo4jLoader
        loader = Neo4jLoader()
        yield HybridRetriever(loader)
        loader.close()

    def test_the_graph_has_the_lane(self, live):
        assert live.lanes("canon_library") == ("", PAGE_LANE), \
            "run tools/build_domain_indexes.py"

    def test_the_page_lane_returns_page_nodes_only(self, live):
        hits = live.fulltext_search("mascarpone", top_k=10, domain="canon_library",
                                    lane=PAGE_LANE)
        assert hits
        props = live.hydrate([i for i, _ in hits])
        assert {p["extraction_method"] for p in props.values()} == {"page"}

    def test_the_lane_puts_pages_in_the_window(self, live):
        """The failure this change exists to fix: with one lane the window was
        90-100% concept nodes."""
        from graphify_ent.embed import Embedder
        enc = Embedder()
        res = live.query("mascarpone eggs sugar tiramisu",
                         embed_fn=lambda q: enc.encode([q])[0],
                         domain="canon_library")
        assert not res.refused
        props = live.hydrate([h.node_id for h in res.hits[:10]])
        pages = [p for p in props.values() if p["extraction_method"] == "page"]
        assert pages, "no page reached the window"


class TestBorrowedTextStaysOutOfTheVector:
    """The overlap exists so a quote straddling a page break can be MATCHED.
    Embedding it would blur two pages into one vector — and would make a node's
    embedding depend on whether the overlap pass had been run, so the same book
    would encode differently on two graphs."""

    def test_the_embedder_cuts_at_the_marker(self):
        from graphify_ent.embed import _node_text
        from graphify_ent.loader import OVERLAP_MARK
        body = "the page's own text" + OVERLAP_MARK.format(page=8) + "the next page"
        out = _node_text({"label": "L", "passage": body})
        assert "the page's own text" in out
        assert "the next page" not in out

    def test_a_passage_without_a_marker_is_untouched(self):
        from graphify_ent.embed import _node_text
        out = _node_text({"label": "L", "passage": "plain page text"})
        assert "plain page text" in out

    def test_the_tool_and_the_package_share_one_marker(self):
        """Two copies of a data convention drift, and the drift is silent."""
        from graphify_ent.loader import OVERLAP_MARK, OVERLAP_PREFIX
        assert OVERLAP_MARK.startswith(OVERLAP_PREFIX)
        assert "OVERLAP_MARK" in ENRICH.read_text()
        assert 'OVERLAP_MARK = ' not in ENRICH.read_text()


class TestFastPathOperatingPoint:
    """The fast path skips the encoder when the corpus literally contains the
    query's words. That was measured on two books, where full term coverage
    meant one passage; on sixteen it means dozens. Measured both ways
    (evidence/T99):

        encoder skipped (default)   canon 95.5%   Q2 page recall 78.9%
        encoder kept                canon 87.0%   Q2 page recall 86.8%

    Neither dominates — the benchmarks ask different things — so the choice is
    exposed instead of decided in secret."""

    def test_the_default_still_skips_the_encoder(self):
        src = (ROOT / "graphify_ent" / "retrieval.py").read_text()
        assert 'os.environ.get("ENTERPRIPHY_FAST_PATH_KEEPS_VECTOR") != "1"' in src
        assert "channels = tuple(c for c in channels if c != \"vector\")" in src

    def test_both_measurements_are_recorded_next_to_the_switch(self):
        """A knob whose trade-off lives only in a commit message is a knob
        nobody can set responsibly."""
        src = (ROOT / "graphify_ent" / "retrieval.py").read_text()
        block = src[src.index("ENTERPRIPHY_FAST_PATH_KEEPS_VECTOR") - 1200:]
        for figure in ("95.5%", "87.0%", "78.9%", "86.8%"):
            assert figure in block, figure
