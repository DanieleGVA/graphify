"""Phase 3.2 — hybrid retrieval pipeline (vector + fulltext + graph, RRF).

Per query (architecture doc §2, execution plan §3.2):

  0. tiered query expansion — deterministic concept/glossary tier first
  1. vector top-k via `db.index.vector.queryNodes`
  2. fulltext top-k via `db.index.fulltext.queryNodes`, per language variant
  3. RRF fusion (k=60) → seeds
  4. 1–2 hop typed graph expansion, weighted by edge confidence and node
     verification (OCR-sourced and `unverified` nodes are down-weighted)
  5. token-budgeted serialization

**Anti-hallucination (Q1 ≥ 99.5 %, architecture-guaranteed).** Two properties
are enforced here, not left to a prompt:

  * every serialized claim carries `evidence` that is a literal substring of the
    node's source text — `verify_evidence_binding()` re-checks this at answer
    time, so an unsupported claim is a machine-detectable defect, not a
    statistic;
  * when fused results are empty or below the support floor, the server returns
    an explicit refusal ("not present in corpus") instead of a synthesized
    answer.

Phase 6.6 adds the bitemporal filter and `as_of` on every tool; the filter is
already threaded through here so that change is additive.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "REFUSAL_TEXT",
    "STRONG_FULLTEXT",
    "balance_by_method",
    "MIN_METHOD_SHARE",
    "RESULT_WINDOW",
    "Hit",
    "RetrievalResult",
    "HybridRetriever",
    "rrf_fuse",
    "serialize_context",
    "verify_evidence_binding",
]

RRF_K = 60
DEFAULT_TOP_K = 25
DEFAULT_SEEDS = 10
DEFAULT_TOKEN_BUDGET = 4_000

# Support floor for the explicit-refusal path. RRF scores are rank-based and
# carry no notion of "relevant at all", so the refusal decision is made on the
# channels' own score semantics *before* fusion.
#
# Calibrated on the pilot (evidence/T32/refusal-calibration.json), not guessed:
#
# Measured over the whole golden set (86 answerable, 3 unanswerable):
#
#   vector best-score   answerable: min 0.687, p5 0.708, p25 0.735, median 0.769
#                       unanswerable: 0.693, 0.704, 0.712
#   fulltext best-score answerable 5.22–9.82, unanswerable 0.0–6.31
#
# Two findings drive the design:
#
# 1. BM25 cannot arbitrate support at all — an out-of-corpus query ("employee
#    stock option vesting schedule") scores 6.31 by matching ordinary English
#    words, above a genuine in-corpus query ("aubergine", 5.22). The *semantic*
#    channel is therefore the arbiter whenever an embedding is available; the
#    lexical floor applies only to lexical-only operation and is explicitly the
#    weaker guarantee.
# 2. The vector bands **overlap**: no threshold both answers everything
#    answerable and refuses everything unanswerable. The floor is a policy
#    choice, measured:
#
#      floor   unanswerable refused   answerable lost
#      0.70    1/3                    3/86
#      0.72    3/3                    6/86   <- chosen
#      0.75    3/3                    29/86
#
# Q1 is the hard gate (blueprint §3): never answering without support outranks
# recall, so 0.72 is the operating point — full refusal coverage at a 7 % recall
# cost, rather than 34 % at 0.75.
MIN_VECTOR_SIMILARITY = 0.72
MIN_FULLTEXT_SCORE = 1.0       # lexical-only fallback; cannot separate on its own

#: The explicit-refusal path. Returning this is a *correct* outcome, never a failure.
REFUSAL_TEXT = "not present in corpus"

#: Lexical score above which a query is considered *lexically settled*: the
#: corpus contains the phrase, so support is already established and the
#: semantic channel has nothing left to decide. Verification queries name a
#: preparation ("Bechamel Sauce", "Sauce Mornay"), which is a lexical problem;
#: skipping the encoder on that path is most of the query's cost. Refusal is
#: unaffected — it is only ever decided on the slow path, where lexical
#: evidence was weak, which is exactly when the semantic channel is needed.
STRONG_FULLTEXT = 6.0

#: How many results a caller is assumed to actually read. Balancing applies to
#: this head of the list only; everything below keeps pure fused order.
RESULT_WINDOW = 10

#: Minimum share of that window guaranteed to each extraction method present.
#: A graph can hold facts extracted in several ways at once — page-sized
#: passages that carry surrounding text, and concept-sized nodes that carry a
#: short quote and the relations around it. They answer different questions,
#: and the window is a fixed budget: measured, 79,658 concept nodes crowded
#: 3,187 passage nodes out of the top 10 and page-level recall fell from 35.2%
#: to 28.3% even as document-level recall rose to 97.2%. Neither method is
#: named here — the rule is that no single one may take the whole window.
MIN_METHOD_SHARE = 0.3


def balance_by_method(
    hits: list["Hit"], window: int = RESULT_WINDOW, min_share: float = MIN_METHOD_SHARE
) -> list["Hit"]:
    """Stop one extraction method from monopolising the returned window.

    Order is otherwise preserved: this caps how many of the top `window` slots
    any one method may take, it does not promote weaker hits above stronger
    ones. With a single method present it is the identity function, so a graph
    built one way behaves exactly as before.
    """
    if window <= 0 or not hits:
        return hits
    methods = {h.extraction_method or "unknown" for h in hits}
    if len(methods) < 2:
        return hits
    quota = max(1, int(window * min_share))
    cap = max(1, window - (len(methods) - 1) * quota)
    counts: dict[str, int] = {}
    head: list[Hit] = []
    tail: list[Hit] = []
    for h in hits:
        m = h.extraction_method or "unknown"
        if len(head) < window and counts.get(m, 0) < cap:
            head.append(h)
            counts[m] = counts.get(m, 0) + 1
        else:
            tail.append(h)
    return head + tail

#: Ranking multipliers — provenance is part of the score, per architecture §2.
CONFIDENCE_WEIGHT = {"EXTRACTED": 1.0, "INFERRED": 0.75, "AMBIGUOUS": 0.5}
METHOD_WEIGHT = {"native": 1.0, "ocr": 0.7}
VERIFICATION_WEIGHT = {"verified": 1.0, None: 1.0, "unverified": 0.6}


@dataclass
class Hit:
    node_id: str
    score: float
    label: str = ""
    source_file: str = ""
    source_location: str | None = None
    text_excerpt: str = ""
    evidence: str = ""
    lang: str | None = None
    confidence: str = "EXTRACTED"
    extraction_method: str = "native"
    verification: str | None = None
    hops: int = 0
    channels: list[str] = field(default_factory=list)

    def provenance_weight(self) -> float:
        return (
            CONFIDENCE_WEIGHT.get(self.confidence, 0.75)
            * METHOD_WEIGHT.get(self.extraction_method, 1.0)
            * VERIFICATION_WEIGHT.get(self.verification, 1.0)
        )


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    refused: bool = False
    refusal_reason: str = ""
    expansions: list[str] = field(default_factory=list)
    channel_counts: dict[str, int] = field(default_factory=dict)
    contradictions: list[tuple[str, str]] = field(default_factory=list)

    @property
    def top_documents(self) -> list[str]:
        seen, out = set(), []
        for h in self.hits:
            if h.source_file not in seen:
                seen.add(h.source_file)
                out.append(h.source_file)
        return out


def rrf_fuse(ranked_lists: dict[str, list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank).

    Rank-based, so it fuses channels whose raw scores are not comparable
    (cosine similarity vs Lucene BM25) without any normalization guesswork.
    """
    scores: dict[str, float] = {}
    for ids in ranked_lists.values():
        for rank, node_id in enumerate(ids, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
    return scores


def _escape_lucene(q: str) -> str:
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", q)


def verify_evidence_binding(text: str, source_text: str, min_len: int = 24) -> bool:
    """True when `text` is genuinely grounded in `source_text`.

    Normalizes whitespace only — the excerpt must otherwise be a literal
    substring. This is the machine check behind Q1: a claim that fails it is a
    defect, not a low score.
    """
    if not text or not source_text:
        return False
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    a, b = norm(text), norm(source_text)
    if len(a) < min_len:
        return a in b
    return a in b


def serialize_context(
    hits: list[Hit], token_budget: int = DEFAULT_TOKEN_BUDGET, chars_per_token: int = 4
) -> str:
    """Token-budgeted context, shaped like `serve.py:_subgraph_to_text`.

    Every block carries its source file, location and evidence excerpt, so an
    answer built from this context can always be traced back to a source node.
    """
    budget_chars = token_budget * chars_per_token
    parts, used = [], 0
    for h in hits:
        loc = f" ({h.source_location})" if h.source_location else ""
        flags = []
        if h.extraction_method == "ocr":
            flags.append("OCR-sourced")
        if h.verification == "unverified":
            flags.append("unverified")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        block = (
            f"### {h.label}{flag_str}\n"
            f"source: {h.source_file}{loc}\n"
            f"evidence: {h.evidence}\n"
        )
        if used + len(block) > budget_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


class HybridRetriever:
    """The retrieval pipeline. Owns no state beyond its Neo4j session factory."""

    def __init__(self, loader, glossary: dict[str, list[str]] | None = None):
        self.loader = loader
        #: language-variant glossary; Phase 6.0 auto-builds this from the
        #: `:Concept` layer's per-language `labels` dict.
        self.glossary = glossary or {}

    # -- tier (a): deterministic expansion ---------------------------------
    def expand_query(self, query: str, deep: bool = False) -> list[str]:
        """Deterministic concept/glossary expansion (tier a).

        Tier (b) LLM rewrite and tier (c) HyDE are deliberately off the fast
        path; they attach here when a model endpoint is configured.
        """
        variants = [query]
        low = query.lower()
        for term, alts in self.glossary.items():
            if term.lower() in low:
                for alt in alts:
                    v = re.sub(re.escape(term), alt, query, flags=re.IGNORECASE)
                    if v not in variants:
                        variants.append(v)
        return variants

    # -- channels ----------------------------------------------------------
    def vector_search(
        self, embedding: list[float], top_k: int = DEFAULT_TOP_K, domain: str | None = None
    ) -> list[tuple[str, float]]:
        cypher = (
            "CALL db.index.vector.queryNodes('entity_embedding', $k, $v) "
            "YIELD node, score "
            "WHERE ($domain IS NULL OR node.domain = $domain) "
            + self._validity_clause("node")
            + " RETURN node.id AS id, score ORDER BY score DESC"
        )
        with self.loader._session() as s:
            return [(r["id"], r["score"]) for r in s.run(cypher, k=top_k, v=embedding,
                                                         domain=domain)]

    def fulltext_search(
        self, text: str, top_k: int = DEFAULT_TOP_K, domain: str | None = None
    ) -> list[tuple[str, float]]:
        cypher = (
            "CALL db.index.fulltext.queryNodes('entity_text', $q) "
            "YIELD node, score "
            "WHERE ($domain IS NULL OR node.domain = $domain) "
            + self._validity_clause("node")
            + " RETURN node.id AS id, score ORDER BY score DESC LIMIT $k"
        )
        with self.loader._session() as s:
            try:
                return [(r["id"], r["score"]) for r in s.run(cypher, q=_escape_lucene(text),
                                                             k=top_k, domain=domain)]
            except Exception:
                return []

    @staticmethod
    def _validity_clause(var: str) -> str:
        """Phase 6.6 default filter: only currently-valid facts.

        Written as a clause the query builder always includes, so enabling
        bitemporal invalidation later is a data change, not a code change.
        """
        return (
            f" AND {var}.invalidated_at IS NULL "
            f"AND ({var}.valid_to IS NULL OR {var}.valid_to > $as_of_default) "
        ).replace("$as_of_default", "datetime()")

    # -- expansion ---------------------------------------------------------
    def expand_graph(self, seed_ids: list[str], hops: int = 1, limit: int = 40) -> list[dict]:
        """1–2 hop typed expansion, weighted by edge confidence."""
        if not seed_ids or hops < 1:
            return []
        cypher = (
            f"MATCH (s:Entity) WHERE s.id IN $ids "
            f"MATCH (s)-[r*1..{min(hops, 2)}]-(n:Entity) "
            f"WHERE NOT n.id IN $ids "
            + self._validity_clause("n")
            + " WITH n, reduce(w = 1.0, x IN r | w * coalesce(x.weight, 1.0)) AS w "
            "RETURN DISTINCT n.id AS id, w ORDER BY w DESC LIMIT $limit"
        )
        with self.loader._session() as s:
            return [dict(r) for r in s.run(cypher, ids=seed_ids, limit=limit)]

    # -- hydration ---------------------------------------------------------
    def hydrate(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        with self.loader._session() as s:
            rows = s.run(
                "MATCH (n:Entity) WHERE n.id IN $ids "
                "RETURN n.id AS id, n.label AS label, n.source_file AS source_file, "
                "n.source_location AS source_location, n.text_excerpt AS text_excerpt, "
                "n.evidence AS evidence, n.lang AS lang, n.confidence AS confidence, "
                "n.extraction_method AS extraction_method, n.verification AS verification",
                ids=node_ids,
            )
            return {r["id"]: dict(r) for r in rows}

    # -- the pipeline ------------------------------------------------------
    def query(
        self,
        query_text: str,
        embedding: list[float] | None = None,
        top_k: int = DEFAULT_TOP_K,
        seeds: int = DEFAULT_SEEDS,
        hops: int = 1,
        domain: str | None = None,
        deep: bool = False,
        channels: tuple[str, ...] = ("vector", "fulltext", "graph"),
        min_support: float = 0.0,
        min_vector_similarity: float = MIN_VECTOR_SIMILARITY,
        min_fulltext_score: float = MIN_FULLTEXT_SCORE,
        balance_methods: bool = True,
        result_window: int = RESULT_WINDOW,
        fast_path: bool = True,
        strong_fulltext: float = STRONG_FULLTEXT,
    ) -> RetrievalResult:
        """Run the hybrid pipeline. `channels` allows the eval's ablations."""
        variants = self.expand_query(query_text, deep=deep)
        ranked: dict[str, list[str]] = {}
        channel_counts: dict[str, int] = {}
        best_vector = 0.0
        best_fulltext = 0.0

        # Fast path: probe the lexical channel first. If the corpus literally
        # contains the phrase, the answer is settled and the embedding — the
        # single most expensive step — is never computed.
        if fast_path and "fulltext" in channels:
            probe = self.fulltext_search(query_text, top_k=top_k, domain=domain)
            if probe:
                best_fulltext = max(s for _, s in probe)
                if best_fulltext >= strong_fulltext:
                    ranked["fulltext:0"] = [i for i, _ in probe]
                    channel_counts["fulltext"] = len(probe)
                    channel_counts["fast_path"] = 1
                    embedding = None
                    channels = tuple(c for c in channels if c != "vector")
                    variants = variants[:1]

        if "vector" in channels and embedding is not None:
            vec = self.vector_search(embedding, top_k=top_k, domain=domain)
            ranked["vector"] = [i for i, _ in vec]
            channel_counts["vector"] = len(vec)
            best_vector = max((s for _, s in vec), default=0.0)

        if "fulltext" in channels and "fulltext:0" not in ranked:
            for idx, variant in enumerate(variants):
                ft = self.fulltext_search(variant, top_k=top_k, domain=domain)
                if ft:
                    ranked[f"fulltext:{idx}"] = [i for i, _ in ft]
                    best_fulltext = max(best_fulltext, max(s for _, s in ft))
            channel_counts["fulltext"] = sum(
                len(v) for k, v in ranked.items() if k.startswith("fulltext")
            )

        # Explicit-refusal path: the corpus does not answer this question, and
        # returning the nearest unrelated passage is precisely the failure Q1
        # exists to prevent. When the semantic channel ran, it decides; lexical
        # score alone is not evidence of support (see the calibration above).
        if channel_counts.get("fast_path"):
            # The phrase is in the corpus; that IS the support.
            supported = True
        elif embedding is not None and "vector" in channels:
            supported = best_vector >= min_vector_similarity
        else:
            supported = best_fulltext >= min_fulltext_score
        if not supported:
            return RetrievalResult(
                query=query_text, hits=[], refused=True,
                refusal_reason=REFUSAL_TEXT, expansions=variants,
                channel_counts=channel_counts,
            )

        fused = rrf_fuse(ranked)
        if not fused:
            return RetrievalResult(
                query=query_text, hits=[], refused=True,
                refusal_reason=REFUSAL_TEXT, expansions=variants,
                channel_counts=channel_counts,
            )

        seed_ids = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:seeds]]

        if "graph" in channels and hops:
            for row in self.expand_graph(seed_ids, hops=hops):
                # Expansion contributes below any directly-retrieved seed.
                fused.setdefault(row["id"], 0.0)
                fused[row["id"]] += 0.5 / (RRF_K + seeds) * float(row.get("w") or 1.0)
            channel_counts["graph"] = len(fused) - len(seed_ids)

        props = self.hydrate(list(fused))
        hits: list[Hit] = []
        for nid, score in sorted(fused.items(), key=lambda kv: -kv[1]):
            p = props.get(nid)
            if not p:
                continue
            hit = Hit(
                node_id=nid,
                score=score,
                label=p.get("label") or "",
                source_file=p.get("source_file") or "",
                source_location=p.get("source_location"),
                text_excerpt=p.get("text_excerpt") or "",
                evidence=p.get("evidence") or "",
                lang=p.get("lang"),
                confidence=p.get("confidence") or "EXTRACTED",
                extraction_method=p.get("extraction_method") or "native",
                verification=p.get("verification"),
                hops=0 if nid in seed_ids else 1,
            )
            hit.score = score * hit.provenance_weight()
            hits.append(hit)

        hits.sort(key=lambda h: -h.score)
        if balance_methods:
            hits = balance_by_method(hits, window=result_window)

        if min_support and (not hits or hits[0].score < min_support):
            return RetrievalResult(
                query=query_text, hits=[], refused=True,
                refusal_reason=REFUSAL_TEXT, expansions=variants,
                channel_counts=channel_counts,
            )

        return RetrievalResult(
            query=query_text, hits=hits, expansions=variants, channel_counts=channel_counts
        )
