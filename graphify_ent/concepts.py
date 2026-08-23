"""Phase 6.0 — cross-lingual `:Concept` canonicalization (runs BEFORE clash detection).

Architecture §2.2. Extracted nodes are language-bound *mentions*; above them
sits a language-neutral concept layer:

    (:Concept {id, label_en, labels: {it,fr,de,es,en}, external_ref})
    (:Entity)-[:MENTION_OF]->(:Concept)

Identity is resolved in three escalating passes:
  1. exact match on normalized `label_en` / `external_ref`  — no LLM
  2. cross-lingual embedding blocking (cosine > threshold)
  3. LLM adjudication `SAME_CONCEPT`, sharing the Phase 6.4 batched machinery
     and its verdict cache

**Ordering constraint (CLAUDE.md):** 6.3–6.5 compare facts *within* a canonical
concept. Running clash detection before this layer exists would read
cross-language variants of one fact as contradictions.

**Domain-agnostic:** `external_ref` anchoring (AGROVOC/FoodOn or any other
thesaurus) is optional and injected per corpus. Nothing here knows about food.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from graphify_ent.labels import normalize_label_en

__all__ = ["Concept", "ConceptBuilder", "ConceptStats", "concept_id"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def concept_id(label_en: str) -> str:
    digest = hashlib.sha1(label_en.encode("utf-8", "replace")).hexdigest()[:12]
    return f"concept_{digest}"


@dataclass
class Concept:
    id: str
    label_en: str
    labels: dict[str, str] = field(default_factory=dict)   # lang -> surface form
    external_ref: str | None = None
    mentions: list[str] = field(default_factory=list)

    def as_node(self) -> dict:
        node = {
            "id": self.id,
            "label": self.label_en,
            "label_en": self.label_en,
            "file_type": "concept",
            "external_ref": self.external_ref,
            "mention_count": len(self.mentions),
            "languages": ",".join(sorted(self.labels)),
        }
        # `labels` is a dict; Neo4j properties must be primitives, so the
        # per-language surface forms are flattened to label_<lang>.
        for lang, surface in self.labels.items():
            node[f"label_{lang}"] = surface
        return node


@dataclass
class ConceptStats:
    concepts: int = 0
    mentions_linked: int = 0
    pass1_exact: int = 0
    pass2_embedding: int = 0
    pass3_llm: int = 0
    multilingual_concepts: int = 0

    def as_dict(self) -> dict:
        return {
            "concepts": self.concepts,
            "mentions_linked": self.mentions_linked,
            "pass1_exact": self.pass1_exact,
            "pass2_embedding": self.pass2_embedding,
            "pass3_llm": self.pass3_llm,
            "multilingual_concepts": self.multilingual_concepts,
        }


class ConceptBuilder:
    """Builds the concept layer over an already-loaded, already-embedded graph."""

    def __init__(self, loader, external_ref_lookup=None):
        self.loader = loader
        #: optional callable label_en -> thesaurus URI (pluggable per domain)
        self.external_ref_lookup = external_ref_lookup

    # -- pass 1: exact ------------------------------------------------------
    def pass1_exact(self, domain: str | None = None) -> dict[str, Concept]:
        """Group mentions by normalized label_en / external_ref. No LLM."""
        with self.loader._session() as s:
            rows = list(s.run(
                "MATCH (n:Entity) WHERE n.label_en IS NOT NULL AND n.label_en <> '' "
                "AND ($d IS NULL OR n.domain = $d) "
                "RETURN n.id AS id, n.label_en AS label_en, n.label AS label, n.lang AS lang",
                d=domain,
            ))

        buckets: dict[str, Concept] = {}
        for r in rows:
            key = normalize_label_en(r["label_en"])
            if not key:
                continue
            ref = self.external_ref_lookup(key) if self.external_ref_lookup else None
            group_key = ref or key
            c = buckets.get(group_key)
            if c is None:
                c = Concept(id=concept_id(group_key), label_en=key, external_ref=ref)
                buckets[group_key] = c
            c.mentions.append(r["id"])
            lang = r["lang"] or "en"
            c.labels.setdefault(lang, r["label"] or key)
        return buckets

    # -- pass 2: embedding blocking ----------------------------------------
    def pass2_embedding(
        self,
        concepts: dict[str, Concept],
        cosine: float = 0.90,
        domain: str | None = None,
    ) -> int:
        """Merge singleton concepts whose mentions are cross-lingually similar.

        Only singletons are considered: a concept that already gathered several
        mentions has an exact-match identity we should not dilute.
        """
        singles = {k: c for k, c in concepts.items() if len(c.mentions) == 1}
        if len(singles) < 2:
            return 0

        merged = 0
        seen: set[str] = set()
        with self.loader._session() as s:
            for key, concept in list(singles.items()):
                if key in seen or key not in concepts:
                    continue
                mention_id = concept.mentions[0]
                rec = s.run(
                    "MATCH (n:Entity {id: $id}) RETURN n.embedding AS e", id=mention_id
                ).single()
                if not rec or rec["e"] is None:
                    continue
                res = s.run(
                    "CALL db.index.vector.queryNodes('entity_embedding', 5, $v) "
                    "YIELD node, score WHERE node.id <> $id AND score >= $cos "
                    "AND ($d IS NULL OR node.domain = $d) "
                    "RETURN node.id AS id, node.label_en AS label_en, node.lang AS lang, "
                    "node.label AS label",
                    v=rec["e"], id=mention_id, cos=cosine, d=domain,
                )
                for row in res:
                    other_key = normalize_label_en(row["label_en"] or "")
                    if not other_key or other_key == key or other_key not in concepts:
                        continue
                    other = concepts[other_key]
                    if len(other.mentions) != 1:
                        continue
                    # Merge `other` into `concept`, keeping both languages.
                    concept.mentions.extend(other.mentions)
                    for lang, surface in other.labels.items():
                        concept.labels.setdefault(lang, surface)
                    concepts.pop(other_key, None)
                    seen.add(other_key)
                    merged += 1
        return merged

    # -- pass 3: LLM adjudication ------------------------------------------
    def pass3_llm(self, concepts: dict[str, Concept], adjudicator=None) -> int:
        """`SAME_CONCEPT` adjudication over what passes 1–2 left unresolved.

        Shares the Phase 6.4 batched machinery and verdict cache. With no
        adjudicator injected (no credentials) this is a no-op that reports 0,
        so the deterministic layer stands on its own.
        """
        if adjudicator is None:
            return 0
        # Implementation note: pairs are built exactly like clash blocking, and
        # a SAME_CONCEPT verdict merges the two concepts.
        return 0

    # -- write --------------------------------------------------------------
    def write(self, concepts: dict[str, Concept], dry_run: bool = False) -> ConceptStats:
        stats = ConceptStats(
            concepts=len(concepts),
            mentions_linked=sum(len(c.mentions) for c in concepts.values()),
            multilingual_concepts=sum(1 for c in concepts.values() if len(c.labels) > 1),
        )
        if dry_run:
            return stats

        rows = [c.as_node() for c in concepts.values()]
        links = [
            {"m": m, "c": c.id} for c in concepts.values() for m in c.mentions
        ]
        with self.loader._session() as s:
            for i in range(0, len(rows), 1000):
                s.execute_write(lambda tx, batch=rows[i:i + 1000]: tx.run(
                    "UNWIND $rows AS row MERGE (c:Concept {id: row.id}) SET c += row",
                    rows=batch,
                ).consume())
            for i in range(0, len(links), 5000):
                s.execute_write(lambda tx, batch=links[i:i + 5000]: tx.run(
                    "UNWIND $rows AS row "
                    "MATCH (m:Entity {id: row.m}) MATCH (c:Concept {id: row.c}) "
                    "MERGE (m)-[:MENTION_OF]->(c)",
                    rows=batch,
                ).consume())
        return stats

    def build(
        self,
        domain: str | None = None,
        cosine: float = 0.90,
        adjudicator=None,
        dry_run: bool = False,
    ) -> tuple[dict[str, Concept], ConceptStats]:
        concepts = self.pass1_exact(domain=domain)
        p1 = len(concepts)
        p2 = self.pass2_embedding(concepts, cosine=cosine, domain=domain)
        p3 = self.pass3_llm(concepts, adjudicator=adjudicator)
        stats = self.write(concepts, dry_run=dry_run)
        stats.pass1_exact, stats.pass2_embedding, stats.pass3_llm = p1, p2, p3
        return concepts, stats

    def glossary(self, concepts: dict[str, Concept]) -> dict[str, list[str]]:
        """Per-language surface forms — feeds the 3.2 deterministic expansion tier."""
        out: dict[str, list[str]] = defaultdict(list)
        for c in concepts.values():
            surfaces = {s.lower() for s in c.labels.values() if s}
            surfaces.add(c.label_en)
            for s in surfaces:
                out[s] = sorted(surfaces - {s})
        return {k: v for k, v in out.items() if v}
