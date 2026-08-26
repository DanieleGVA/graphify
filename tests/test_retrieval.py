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
    content_query,
    content_terms,
    rrf_fuse,
    serialize_context,
    verify_evidence_binding,
    balance_by_method,
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


class TestContentTerms:
    """A question is mostly scaffolding; scoring the scaffolding is how BM25
    ranked a page about flour above the page that answers the question."""

    def test_drops_interrogatives_and_auxiliaries(self):
        assert content_terms("What does HACCP stand for?") == ["HACCP", "stand"]

    def test_keeps_figures_and_units(self):
        terms = content_terms("How much sugar for 8 oz/227 g of egg whites?")
        assert terms == ["sugar", "8", "227", "egg", "whites"], \
            "figures survive; the slash is a separator, not part of a token"

    def test_handles_french_and_italian_questions(self):
        assert content_terms("Quelles sont les proportions d'un roux ?") == \
            ["proportions", "roux"]
        assert content_terms("Che proporzioni ha un roux tra farina e burro?") == \
            ["proporzioni", "roux", "farina", "burro"]

    def test_never_returns_empty_query(self, monkeypatch):
        """All-stopword input must fall back to the original: an empty Lucene
        query returns nothing at all, which reads as 'corpus does not know'."""
        monkeypatch.setenv("ENTERPRIPHY_CONTENT_FILTER", "1")
        assert content_query("what is it?") == "what is it?"

    def test_deduplicates_while_preserving_order(self):
        assert content_terms("sugar and sugar and salt") == ["sugar", "salt"]

    def test_reduction_is_off_by_default(self, monkeypatch):
        """Measured: reducing the BM25 query helps a weak encoder and hurts a
        strong one (Q2 69.0% reduced vs 71.4% whole, same encoder and floor)."""
        monkeypatch.delenv("ENTERPRIPHY_CONTENT_FILTER", raising=False)
        assert content_query("What does HACCP stand for?") == "What does HACCP stand for?"

    def test_reduction_can_be_switched_on_for_ablation(self, monkeypatch):
        monkeypatch.setenv("ENTERPRIPHY_CONTENT_FILTER", "1")
        assert content_query("What does HACCP stand for?") == "HACCP stand"


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
        """The floor must not suppress genuine answers.

        Precondition: the corpus must be fully embedded. A partially-embedded
        database legitimately has no high-similarity node yet, which would make
        this fail for a reason that has nothing to do with the floor.
        """
        from graphify_ent.embed import Embedder

        with retriever.loader._session() as s:
            pending = s.run(
                "MATCH (n:Entity) WHERE n.embedding IS NULL RETURN count(n) AS c"
            ).single()["c"]
        if pending:
            pytest.skip(f"corpus not fully embedded ({pending} nodes pending)")

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


class TestBalanceByMethod:
    """One extraction method must not take the whole result window.

    Measured motivation: with 79,658 concept nodes and 3,187 passage nodes in
    one graph, the concept nodes took all ten top slots and page-level recall
    fell from 35.2% to 28.3% even though document-level recall rose to 97.2%.
    """

    @staticmethod
    def _hit(i: int, method: str) -> Hit:
        return Hit(node_id=f"n{i}", score=1000.0 - i, label=f"l{i}",
                   source_file="book.pdf", extraction_method=method)

    def test_single_method_is_untouched(self):
        hits = [self._hit(i, "native") for i in range(15)]
        assert balance_by_method(hits) == hits

    def test_empty_is_untouched(self):
        assert balance_by_method([]) == []

    def test_minority_method_gets_its_share_of_the_window(self):
        hits = ([self._hit(i, "llm") for i in range(30)]
                + [self._hit(100 + i, "native") for i in range(5)])
        hits.sort(key=lambda h: -h.score)
        assert all(h.extraction_method == "llm" for h in hits[:10])

        out = balance_by_method(hits, window=10, min_share=0.3)
        methods = [h.extraction_method for h in out[:10]]
        assert methods.count("native") >= 3
        assert methods.count("llm") == 7

    def test_nothing_is_dropped_or_duplicated(self):
        hits = ([self._hit(i, "llm") for i in range(20)]
                + [self._hit(100 + i, "native") for i in range(20)])
        out = balance_by_method(hits)
        assert len(out) == len(hits)
        assert {h.node_id for h in out} == {h.node_id for h in hits}

    def test_order_within_a_method_is_preserved(self):
        """Capping must not promote a weaker hit above a stronger one of the
        same method — it only limits how many slots that method may take."""
        hits = ([self._hit(i, "llm") for i in range(20)]
                + [self._hit(100 + i, "native") for i in range(20)])
        hits.sort(key=lambda h: -h.score)
        out = balance_by_method(hits)
        for method in ("llm", "native"):
            before = [h.node_id for h in hits if h.extraction_method == method]
            after = [h.node_id for h in out if h.extraction_method == method]
            assert before == after

    def test_minority_scarcity_does_not_waste_slots(self):
        """If the minority method has only one candidate, the window must still
        be filled — a guaranteed share is a ceiling on the majority, not a hole."""
        hits = ([self._hit(i, "llm") for i in range(20)]
                + [self._hit(100, "native")])
        hits.sort(key=lambda h: -h.score)
        out = balance_by_method(hits, window=10, min_share=0.3)
        assert len(out[:10]) == 10
        assert sum(1 for h in out[:10] if h.extraction_method == "native") == 1


class TestLexicalSupport:
    """Lexical evidence means a passage containing most of the query, not a
    high BM25 score.

    Measured regression: "employee stock option vesting schedule" matched
    "stock" — the broth, not the share — scored above the floor, and was
    answered instead of refused. Coverage is what separates the two.
    """

    def test_coverage_floor_is_a_fraction_of_the_query(self):
        from graphify_ent.retrieval import MIN_TERM_COVERAGE
        assert 0.0 < MIN_TERM_COVERAGE <= 1.0

    def test_required_terms_query_is_a_conjunction(self):
        q = HybridRetriever._lucene_all("Mornay Sauce Gruyere")
        assert q.count("+") == 3
        assert "mornay" in q.lower()

    def test_short_words_do_not_become_requirements(self):
        """"and", "of" carry no information and would only narrow wrongly."""
        q = HybridRetriever._lucene_all("roux and flour")
        assert "+and" not in q.lower()

    def test_hydrate_factor_covers_more_than_the_window(self):
        """Provenance weighting reorders hits, so hydrating exactly the window
        would drop candidates that were about to be promoted into it."""
        from graphify_ent.retrieval import HYDRATE_FACTOR
        assert HYDRATE_FACTOR >= 2
