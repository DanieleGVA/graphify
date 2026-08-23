"""Phase 3.2 — hybrid retrieval + the anti-hallucination guarantees.

The Q1 controls (evidence binding, explicit refusal) are tested as *behaviour*,
not as prompt wording: they are the properties the architecture claims are
guaranteed by construction.
"""

from __future__ import annotations

import os

import pytest

from graphify_ent.retrieval import (
    REFUSAL_TEXT,
    Hit,
    HybridRetriever,
    rrf_fuse,
    serialize_context,
    verify_evidence_binding,
)

NEO4J_URI = os.environ.get("NEO4J_URI")
requires_neo4j = pytest.mark.skipif(not NEO4J_URI, reason="NEO4J_URI not set")


class TestRRF:
    def test_fuses_by_rank_not_score(self):
        fused = rrf_fuse({"a": ["x", "y"], "b": ["y", "x"]})
        assert round(fused["x"], 9) == round(fused["y"], 9), "symmetric ranks tie"

    def test_agreement_across_channels_wins(self):
        fused = rrf_fuse({"vec": ["shared", "only_vec"], "ft": ["shared", "only_ft"]})
        assert fused["shared"] > fused["only_vec"]
        assert fused["shared"] > fused["only_ft"]

    def test_rank_one_beats_rank_two(self):
        fused = rrf_fuse({"c": ["first", "second"]})
        assert fused["first"] > fused["second"]

    def test_empty_input(self):
        assert rrf_fuse({}) == {}


class TestEvidenceBinding:
    """Q1's machine check: an unsupported claim is a defect, not a statistic."""

    def test_literal_substring_passes(self):
        src = "Bring the stock to a simmer and skim the surface carefully."
        assert verify_evidence_binding("simmer and skim the surface", src) is True

    def test_whitespace_differences_are_tolerated(self):
        src = "Bring the stock to a simmer\nand   skim the surface."
        assert verify_evidence_binding("simmer and skim the surface", src) is True

    def test_fabricated_claim_fails(self):
        src = "Bring the stock to a simmer and skim the surface carefully."
        assert verify_evidence_binding("add 200g of saffron and refrigerate", src) is False

    def test_paraphrase_fails(self):
        """Paraphrase is exactly what evidence binding must reject."""
        src = "Bring the stock to a simmer and skim the surface carefully."
        assert verify_evidence_binding("heat the broth gently and remove foam", src) is False

    def test_empty_inputs_fail_closed(self):
        assert verify_evidence_binding("", "something") is False
        assert verify_evidence_binding("something", "") is False


class TestSerialization:
    def _hits(self, n=3, **kw):
        return [
            Hit(node_id=f"n{i}", score=1.0 - i / 10, label=f"Node {i}",
                source_file="book.pdf", source_location=f"pages {i}-{i+1}",
                evidence=f"evidence text {i}", **kw)
            for i in range(n)
        ]

    def test_every_block_carries_source_and_evidence(self):
        out = serialize_context(self._hits())
        for i in range(3):
            assert f"evidence text {i}" in out
        assert out.count("source: book.pdf") == 3

    def test_token_budget_truncates_rather_than_overflowing(self):
        long_hits = [
            Hit(node_id=f"n{i}", score=1.0, label="L", source_file="b.pdf",
                evidence="x" * 500)
            for i in range(50)
        ]
        out = serialize_context(long_hits, token_budget=200)
        assert len(out) <= 200 * 4 + 600

    def test_ocr_and_unverified_are_disclosed(self):
        hits = [Hit(node_id="a", score=1, label="A", source_file="s.pdf",
                    evidence="e", extraction_method="ocr", verification="unverified")]
        out = serialize_context(hits)
        assert "OCR-sourced" in out and "unverified" in out

    def test_empty_hits_serialize_to_empty(self):
        assert serialize_context([]) == ""


class TestProvenanceWeighting:
    def test_ocr_is_down_weighted(self):
        native = Hit("a", 1.0, extraction_method="native")
        ocr = Hit("b", 1.0, extraction_method="ocr")
        assert ocr.provenance_weight() < native.provenance_weight()

    def test_inferred_below_extracted(self):
        assert Hit("a", 1, confidence="INFERRED").provenance_weight() < \
               Hit("b", 1, confidence="EXTRACTED").provenance_weight()

    def test_unverified_is_down_weighted(self):
        assert Hit("a", 1, verification="unverified").provenance_weight() < \
               Hit("b", 1, verification="verified").provenance_weight()


class TestQueryExpansion:
    def test_glossary_produces_language_variants(self):
        r = HybridRetriever(loader=None, glossary={"eggplant": ["aubergine", "melanzana"]})
        variants = r.expand_query("how to roast eggplant")
        assert "how to roast aubergine" in variants
        assert "how to roast melanzana" in variants
        assert variants[0] == "how to roast eggplant", "original must stay first"

    def test_no_glossary_match_is_a_passthrough(self):
        r = HybridRetriever(loader=None, glossary={"eggplant": ["aubergine"]})
        assert r.expand_query("braised beef") == ["braised beef"]


@requires_neo4j
class TestAgainstLiveGraph:
    @pytest.fixture(scope="class")
    def retriever(self):
        from graphify_ent.loader import Neo4jLoader

        loader = Neo4jLoader()
        yield HybridRetriever(loader)
        loader.close()

    def test_fulltext_returns_hits_from_the_pilot(self, retriever):
        hits = retriever.fulltext_search("sauce", top_k=10, domain="pilot")
        if not hits:
            pytest.skip("pilot graph not loaded in this database")
        assert len(hits) > 0

    def test_unanswerable_query_refuses(self, retriever):
        """The explicit-refusal path must fire rather than return near neighbours."""
        from graphify_ent.embed import Embedder

        q = "kubernetes ingress controller misconfiguration"
        try:
            emb = Embedder().encode([q])[0]
        except Exception:
            emb = None
        res = retriever.query(q, embedding=emb,
                              channels=("vector", "fulltext"), hops=0, domain="pilot")
        assert res.refused, "out-of-corpus query must be refused, not answered"
        assert res.refusal_reason == REFUSAL_TEXT
        assert res.hits == []

    def test_in_corpus_query_is_not_refused(self, retriever):
        """The floor must not suppress genuine answers."""
        from graphify_ent.embed import Embedder

        q = "how to prepare a classic sauce"
        try:
            emb = Embedder().encode([q])[0]
        except Exception:
            pytest.skip("embedder unavailable")
        res = retriever.query(q, embedding=emb,
                              channels=("vector", "fulltext"), hops=0, domain="pilot")
        if not res.hits and not res.refused:
            pytest.skip("pilot graph not loaded")
        assert not res.refused, "a corpus-answerable query must not be refused"

    def test_every_hit_is_evidence_bound(self, retriever):
        res = retriever.query("sauce", embedding=None, channels=("fulltext",),
                              hops=0, domain="pilot")
        if not res.hits:
            pytest.skip("pilot graph not loaded")
        props = retriever.hydrate([h.node_id for h in res.hits[:10]])
        for h in res.hits[:10]:
            src = (props.get(h.node_id) or {}).get("text_excerpt") or ""
            assert verify_evidence_binding(h.evidence, src), f"{h.node_id} not evidence-bound"
