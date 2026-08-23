"""Phase 6 — canonicalization, bitemporal invalidation, clash detection.

Acceptance (execution plan §6): ≥8/10 planted contradictions detected; zero
silent resolutions; planted supersessions ordered by doc_date; `as_of` returns
the pre-supersession state.
"""

from __future__ import annotations

import json
import os

import pytest

from graphify_ent.clash import (
    ClashEngine,
    ClashPair,
    Verdict,
    VerdictCache,
    pair_content_hash,
    resolve,
)
from graphify_ent.concepts import Concept, ConceptBuilder, concept_id
from graphify_ent.temporal import (
    ManifestDiff,
    TemporalEngine,
    build_manifest,
    diff_manifests,
)

NEO4J_URI = os.environ.get("NEO4J_URI")
requires_neo4j = pytest.mark.skipif(not NEO4J_URI, reason="NEO4J_URI not set")


# ------------------------------------------------------------ 6.1 manifests

class TestManifestDiff:
    def test_detects_change_delete_add(self):
        old = {"a.pdf": "h1", "b.pdf": "h2", "c.pdf": "h3"}
        new = {"a.pdf": "h1", "b.pdf": "CHANGED", "d.pdf": "h4"}
        d = diff_manifests(old, new)
        assert d.changed == ["b.pdf"]
        assert d.deleted == ["c.pdf"]
        assert d.added == ["d.pdf"]

    def test_invalidating_is_changed_plus_deleted(self):
        d = ManifestDiff(changed=["b.pdf"], deleted=["c.pdf"], added=["d.pdf"])
        assert d.invalidating == ["b.pdf", "c.pdf"]

    def test_identical_manifests_invalidate_nothing(self):
        m = {"a.pdf": "h1"}
        assert diff_manifests(m, m).invalidating == []

    def test_build_manifest_hashes_real_files(self, tmp_path):
        (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4 content")
        m = build_manifest(tmp_path)
        assert set(m) == {"x.pdf"} and len(m["x.pdf"]) == 64


# -------------------------------------------------------- 6.5 resolution

class TestResolutionChain:
    """Zero silent winners: every auto-resolution records which policy decided."""

    def _pair(self):
        return ClashPair("a", "b", "text a", "text b", 0.91)

    def test_authority_wins_first(self):
        props = {"a": {"source_rank": 1, "doc_date": "2020-01-01", "doc_date_confidence": 0.9},
                 "b": {"source_rank": 3, "doc_date": "2026-01-01", "doc_date_confidence": 0.9}}
        r = resolve(self._pair(), Verdict("CONTRADICTORY"), props)
        assert r.action == "supersede" and r.winner == "a"
        assert r.policy == "authority"
        assert r.rationale, "an auto-resolution must never be silent"

    def test_recency_breaks_authority_ties(self):
        props = {"a": {"source_rank": 2, "doc_date": "2020-01-01", "doc_date_confidence": 0.9},
                 "b": {"source_rank": 2, "doc_date": "2026-01-01", "doc_date_confidence": 0.9}}
        r = resolve(self._pair(), Verdict("CONTRADICTORY"), props)
        assert r.winner == "b" and r.policy == "recency"

    def test_low_date_confidence_does_not_decide(self):
        props = {"a": {"source_rank": 2, "doc_date": "2020-01-01", "doc_date_confidence": 0.4},
                 "b": {"source_rank": 2, "doc_date": "2026-01-01", "doc_date_confidence": 0.4}}
        r = resolve(self._pair(), Verdict("CONTRADICTORY"), props)
        assert r.policy != "recency"

    def test_confidence_is_the_last_tiebreak(self):
        props = {"a": {"source_rank": 2, "confidence": "EXTRACTED"},
                 "b": {"source_rank": 2, "confidence": "AMBIGUOUS"}}
        r = resolve(self._pair(), Verdict("CONTRADICTORY"), props)
        assert r.winner == "a" and r.policy == "confidence"

    def test_ocr_is_down_ranked(self):
        props = {"a": {"source_rank": 2, "confidence": "EXTRACTED", "extraction_method": "ocr"},
                 "b": {"source_rank": 2, "confidence": "EXTRACTED", "extraction_method": "native"}}
        r = resolve(self._pair(), Verdict("CONTRADICTORY"), props)
        assert r.winner == "b"

    def test_unresolvable_becomes_contradiction_not_a_guess(self):
        props = {"a": {"source_rank": 2, "confidence": "EXTRACTED"},
                 "b": {"source_rank": 2, "confidence": "EXTRACTED"}}
        r = resolve(self._pair(), Verdict("CONTRADICTORY"), props)
        assert r.action == "contradict"
        assert r.winner is None, "never pick a side silently"

    def test_same_and_complementary_are_not_conflicts(self):
        for v in ("SAME", "COMPLEMENTARY"):
            r = resolve(self._pair(), Verdict(v), {})
            assert r.action == "none"

    def test_adjudicated_supersedes_is_honoured(self):
        v = Verdict("SUPERSEDES", winner="b", rationale="b explicitly replaces a")
        r = resolve(self._pair(), v, {})
        assert r.action == "supersede" and r.winner == "b"
        assert r.policy == "adjudicated-supersedes"


# ------------------------------------------------------- 6.4 verdict cache

class TestVerdictCache:
    def test_hash_is_order_independent(self):
        p1 = ClashPair("a", "b", "TA", "TB")
        p2 = ClashPair("b", "a", "TB", "TA")
        assert pair_content_hash(p1) == pair_content_hash(p2)

    def test_hash_changes_with_content(self):
        assert pair_content_hash(ClashPair("a", "b", "X", "Y")) != \
               pair_content_hash(ClashPair("a", "b", "X", "Z"))

    def test_cache_roundtrip_and_persistence(self, tmp_path):
        c = VerdictCache(tmp_path / "v.json")
        c.put("k", Verdict("CONTRADICTORY", rationale="r", confidence=0.9))
        c.flush()
        again = VerdictCache(tmp_path / "v.json")
        got = again.get("k")
        assert got and got.verdict == "CONTRADICTORY" and got.confidence == 0.9

    def test_unchanged_pairs_are_never_re_adjudicated(self, tmp_path):
        """The cost control: adjudication must not repeat on identical content."""
        calls = {"n": 0}

        def adjudicator(pairs):
            calls["n"] += len(pairs)
            return [Verdict("SAME", rationale="same") for _ in pairs]

        engine = ClashEngine(loader=None, cache=VerdictCache(tmp_path / "v.json"))
        pairs = [ClashPair("a", "b", "TA", "TB"), ClashPair("c", "d", "TC", "TD")]

        engine.adjudicate(pairs, adjudicator)
        assert calls["n"] == 2
        engine.adjudicate(pairs, adjudicator)
        assert calls["n"] == 2, "second pass must be served entirely from cache"

    def test_invalid_verdict_is_coerced_not_trusted(self, tmp_path):
        engine = ClashEngine(loader=None, cache=VerdictCache(tmp_path / "v.json"))
        judged = engine.adjudicate(
            [ClashPair("a", "b", "TA", "TB")],
            lambda pairs: [Verdict("NONSENSE")],
        )
        assert judged[0][1].verdict in ("SAME", "COMPLEMENTARY", "CONTRADICTORY", "SUPERSEDES")

    def test_contradiction_without_both_excerpts_loses_confidence(self, tmp_path):
        """Evidence-binding discipline: a verdict must quote both sources."""
        engine = ClashEngine(loader=None, cache=VerdictCache(tmp_path / "v.json"))
        judged = engine.adjudicate(
            [ClashPair("a", "b", "TA", "TB")],
            lambda pairs: [Verdict("CONTRADICTORY", confidence=0.99)],
        )
        assert judged[0][1].confidence <= 0.5


# ------------------------------------------------------------ 6.0 concepts

class TestConceptLayer:
    def test_concept_id_is_stable(self):
        assert concept_id("eggplant") == concept_id("eggplant")
        assert concept_id("eggplant") != concept_id("courgette")

    def test_concept_node_flattens_language_labels(self):
        c = Concept(id="c1", label_en="eggplant",
                    labels={"it": "melanzana", "fr": "aubergine"}, mentions=["m1", "m2"])
        node = c.as_node()
        assert node["label_it"] == "melanzana" and node["label_fr"] == "aubergine"
        assert node["mention_count"] == 2
        assert "it" in node["languages"] and "fr" in node["languages"]

    def test_glossary_is_bidirectional(self):
        b = ConceptBuilder(loader=None)
        gloss = b.glossary({
            "eggplant": Concept("c1", "eggplant",
                                labels={"it": "melanzana", "fr": "aubergine"})
        })
        assert "melanzana" in gloss["aubergine"]
        assert "aubergine" in gloss["melanzana"]


# ------------------------------------------------- integration (live Neo4j)

@requires_neo4j
class TestPhase6Integration:
    @pytest.fixture
    def seeded(self, tmp_path):
        """A small planted corpus: a v1/v2 policy pair plus a contradiction."""
        from graphify_ent.loader import Neo4jLoader

        loader = Neo4jLoader()
        loader.apply_schema()
        # Clean only this test's own domain. A fixture that wipes a shared
        # database destroys any corpus loaded alongside it (it ate the pilot
        # graph and its embeddings once already).
        with loader._session() as s:
            s.run("MATCH (n:Entity {domain:'phase6test'}) DETACH DELETE n")
            s.run("MATCH (c:Concept) WHERE c.mention_count IS NOT NULL "
                  "AND NOT (c)<-[:MENTION_OF]-() DETACH DELETE c")
        nodes = [
            {"id": "p1", "label": "Storage policy", "label_en": "storage policy",
             "file_type": "document", "source_file": "policy_v1.pdf", "version": 1,
             "doc_date": "2024-01-01", "doc_date_confidence": 0.9, "source_rank": 2,
             "confidence": "EXTRACTED", "text_excerpt": "Store at 4 degrees",
             "evidence": "Store at 4 degrees"},
            {"id": "p2", "label": "Storage policy", "label_en": "storage policy",
             "file_type": "document", "source_file": "policy_v2.pdf", "version": 2,
             "doc_date": "2026-01-01", "doc_date_confidence": 0.9, "source_rank": 2,
             "confidence": "EXTRACTED", "text_excerpt": "Store at 2 degrees",
             "evidence": "Store at 2 degrees"},
        ]
        p = tmp_path / "g.json"
        p.write_text(json.dumps({"nodes": nodes, "edges": []}))
        loader.load(p, domain="phase6test")
        yield loader
        with loader._session() as s:
            s.run("MATCH (n:Entity {domain:'phase6test'}) DETACH DELETE n")
        loader.close()

    def test_file_level_invalidation_never_deletes(self, seeded):
        eng = TemporalEngine(seeded)
        before = seeded.count_where("n.domain = 'phase6test'")
        rep = eng.invalidate_files(["policy_v1.pdf"], domain="phase6test")
        assert rep.invalidated_nodes == 1
        assert seeded.count_where("n.domain = 'phase6test'") == before, \
            "invalidation must never delete"

    def test_as_of_returns_pre_supersession_state(self, seeded):
        eng = TemporalEngine(seeded)
        eng.invalidate_files(["policy_v1.pdf"], domain="phase6test")
        now_valid = eng.valid_nodes(domain="phase6test")
        past_valid = eng.valid_nodes(as_of="2025-01-01T00:00:00Z", domain="phase6test")
        assert now_valid == 1
        assert past_valid == 2, "time travel must see the pre-invalidation world"

    def test_version_supersession_orders_by_version(self, seeded):
        eng = TemporalEngine(seeded)
        rep = eng.apply_version_supersession(domain="phase6test")
        assert rep.version_supersessions >= 1
        with seeded._session() as s:
            rec = s.run(
                "MATCH (o:Entity {id:'p1'})-[r:SUPERSEDED_BY]->(n:Entity {id:'p2'}) "
                "RETURN r.policy AS policy, o.valid_to AS valid_to"
            ).single()
        assert rec and rec["policy"] == "version-level"
        assert rec["valid_to"] == "2026-01-01", "valid_to = newer.doc_date"

    def test_concept_layer_groups_by_label_en(self, seeded):
        builder = ConceptBuilder(seeded)
        concepts, stats = builder.build(domain="phase6test")
        assert stats.concepts >= 1
        assert stats.mentions_linked == 2, "both mentions link to one concept"
        with seeded._session() as s:
            c = s.run(
                "MATCH (m:Entity {domain:'phase6test'})-[:MENTION_OF]->(c:Concept) "
                "RETURN count(DISTINCT c) AS c"
            ).single()["c"]
        assert c == 1
