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
from typing import Any, Callable

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
# Re-measured on the current corpus and encoder
# (evidence/T73/refusal-calibration.json). The floor is a property of BOTH, and
# carrying over a number calibrated for a different model is how a system stops
# refusing without anyone noticing:
#
#   in corpus      min 0.736   p5 0.760   median 0.825   max 0.893   (n=34)
#   out of corpus  min 0.560              median 0.622   max 0.660   (n=20)
#
#     floor   out-of-corpus refused   in-corpus lost
#     0.60    6/20                    0/34
#     0.65    18/20                   0/34
#     0.70    20/20                   0/34   <- chosen, sits in the gap
#     0.75    20/20                   1/34
#
# The bands SEPARATE here, which they did not on the six-book corpus under
# BGE-m3 (there they overlapped, and full refusal coverage cost 56% of recall).
# Two things changed: MiniLM discriminates better on this material, and a single
# coherent document has no near-neighbour for an unrelated question. 0.70 sits
# in the gap with margin on both sides rather than on either boundary.
#
# BM25 still cannot arbitrate support: an out-of-corpus query scores high on a
# rare word precisely BECAUSE it is rare. The lexical channel establishes
# support only when a single node contains EVERY term of the query.
MIN_VECTOR_SIMILARITY = 0.70
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

#: Fraction of a query's terms one passage must contain before it counts as
#: lexical evidence. See the measurement in `_lexical_evidence`.
MIN_TERM_COVERAGE = 0.6

#: How many times the result window is hydrated. Provenance weighting can
#: reorder hits, so the window's worth of candidates is not enough on its own;
#: three times it is, and still a fraction of a large fused set.
HYDRATE_FACTOR = 3

#: Function words carry no evidence. Counting them toward coverage makes the
#: threshold easier to clear precisely because they appear everywhere, which
#: weakens the guard they are measured against; requiring them narrows on
#: nothing. Three languages, because the corpus can hold three.
_FUNCTION_WORDS = frozenset("""
and the for with from that this into out are was were has have had not but
les des une aux par sur dans pour avec est sont ont été plus
del della delle degli con per una sono stato come anche
what which who whom whose when where why how does did doing must should would
could shall will can may might there their they them its it's you your
quel quelle quels quelles quoi comment combien pourquoi lorsque dont leur leurs
faut fait font doit doivent peut peuvent sera seront
quale quali quanto quanta quanti quante cosa perche perché quando dove
che chi cui tra fra much many
deve devono viene vengono vanno prima dopo essere stata state stati
""".split())
#: Interrogative and auxiliary words were absent from the set above until Q2
#: was first measured (2026-08-26). They mattered: "What does HACCP stand for?"
#: retrieved Flour, gluten and Stabilizers, while "HACCP stands for" retrieved
#: the right page first. Every earlier benchmark had queried in keyword form,
#: so the whole system had been tuned and measured without ever seeing a
#: question. Removing a word here only stops it being REQUIRED — it never
#: excludes a passage — so the set errs on the generous side.

#: Coverage at which a query counts as settled by one passage, so graph
#: expansion is skipped. Above this the neighbours are noise, and the hop is
#: pure latency on the path that runs most often.
SETTLED_COVERAGE = 0.85

#: How many lexical candidates are scanned for term coverage. Bounded because
#: the scan is a string search per candidate: unbounded, it ran over every OR
#: match in the corpus and cost more than it saved.
LEXICAL_PROBE = 60

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


def content_terms(text: str) -> list[str]:
    """The words of a query that carry meaning, in order, duplicates dropped.

    A question is mostly scaffolding: of "What does HACCP stand for?" only two
    words select anything. Scoring the scaffolding is how BM25 ranked a page
    about flour above the page that answers it.
    """
    seen, out = set(), []
    for w in re.findall(r"[\wÀ-ſ°]+", text or ""):
        low = w.lower()
        if len(low) < 3 and not any(c.isdigit() for c in low):
            continue
        if low in _FUNCTION_WORDS or low in seen:
            continue
        seen.add(low)
        out.append(w)
    return out


def content_query(text: str) -> str:
    """`text` reduced to its content words; the original if nothing survives."""
    terms = content_terms(text)
    return " ".join(terms) if terms else text


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
        #: term -> does this corpus contain it. See `_known_terms`.
        self._vocab: dict[tuple[str, str | None], int] = {}

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
            try:
                return [(r["id"], r["score"]) for r in s.run(cypher, k=top_k, v=embedding,
                                                             domain=domain)]
            except Exception as exc:
                # A width mismatch between the encoder and the index is a
                # misconfiguration, not a query failure: say so plainly rather
                # than surface a Java stack trace from three layers down.
                if "dimensions" in str(exc):
                    raise RuntimeError(
                        f"embedding width {len(embedding)} does not match the vector "
                        f"index; set EMBED_MODEL/EMBED_DIM consistently, or rebuild "
                        f"the index after changing model ({exc})") from exc
                raise

    def _lexical_evidence(self, query_text: str, domain: str | None, top_k: int,
                          must: str = "") -> tuple[list[tuple[str, float]], float]:
        """Candidates that literally contain most of the query, and how much.

        One round trip. The term counting happens in the database, so what
        crosses the wire is a number per candidate rather than forty passages —
        measured, shipping the passages back to count them in Python was the
        largest remaining cost in a query.

        Coverage, not score, is what establishes lexical support. A BM25 score
        is high for a rare word precisely BECAUSE it is rare: "employee stock
        option vesting schedule" matched "stock" — the broth — and was answered
        instead of refused. One passage containing most of what was asked is
        evidence; one passage containing one word of it is not.
        """
        terms = sorted({w.lower() for w in re.findall(r"[\w\u00c0-\u017f]{3,}", query_text)}
                       - _FUNCTION_WORDS)
        if not terms:
            return [], 0.0
        anchors = sorted({w.lower() for w in re.findall(r"[\w\u00c0-\u017f]{3,}", must)}
                         - _FUNCTION_WORDS)
        cypher = (
            # Cut to the best candidates BEFORE the text scan. Without this the
            # CONTAINS ran over every OR match in the corpus and the "one round
            # trip" version came out slower than the two-query one it replaced.
            "CALL db.index.fulltext.queryNodes('entity_text', $q) YIELD node, score "
            "WITH node, score ORDER BY score DESC LIMIT $probe "
            "WHERE ($domain IS NULL OR node.domain = $domain) "
            + self._validity_clause("node") +
            " WITH node, score, toLower(coalesce(node.passage,'') + ' ' + "
            "coalesce(node.label,'') + ' ' + coalesce(node.text_excerpt,'')) AS body "
            "WITH node, score, body, "
            "size([t IN $terms WHERE body CONTAINS t]) AS hits, "
            "size([a IN $anchors WHERE body CONTAINS a]) AS anchored "
            "WHERE anchored = size($anchors) "
            "RETURN node.id AS id, score, hits ORDER BY hits DESC, score DESC LIMIT $k")
        with self.loader._session() as s:
            try:
                rows = list(s.run(cypher, q=_escape_lucene(content_query(query_text)),
                                  domain=domain,
                                  terms=terms, anchors=anchors, k=max(top_k, 25),
                                  probe=LEXICAL_PROBE))
            except Exception:
                return [], 0.0
        if not rows:
            return [], 0.0
        best = rows[0]["hits"] / len(terms)
        return [(r["id"], r["score"]) for r in rows], best

    @staticmethod
    def _lucene_all(text: str, keep: int = 0) -> str:
        """Require every meaningful term instead of accepting any of them.

        Default OR semantics sink the answer: "Mornay Sauce Gruyere Parmesan"
        matched hundreds of nodes on the word "sauce" alone, and the eleven
        nodes that actually say Mornay never reached the top 25. Requiring the
        terms makes the rare one decide the result, which is what a
        verification query means.
        """
        terms = [w for w in re.findall(r"[\w\u00c0-\u017f]{3,}", text)
                 if w.lower() not in _FUNCTION_WORDS]
        if keep and len(terms) > keep:
            # Graded relaxation: keep the most specific terms, longest first.
            # Length is a crude proxy for specificity, but a domain-agnostic
            # one — no word list to maintain per corpus.
            terms = sorted(terms, key=len, reverse=True)[:keep]
        return " ".join(f"+{_escape_lucene(w)}" for w in terms)

    def fulltext_search(
        self, text: str, top_k: int = DEFAULT_TOP_K, domain: str | None = None,
        require_all: bool = False, require_terms: int = 0,
    ) -> list[tuple[str, float]]:
        cypher = (
            "CALL db.index.fulltext.queryNodes('entity_text', $q) "
            "YIELD node, score "
            "WHERE ($domain IS NULL OR node.domain = $domain) "
            + self._validity_clause("node")
            + " RETURN node.id AS id, score ORDER BY score DESC LIMIT $k"
        )
        query = (self._lucene_all(text, keep=require_terms) if require_all
                 else _escape_lucene(content_query(text)))
        if not query.strip():
            return []
        with self.loader._session() as s:
            try:
                return [(r["id"], r["score"])
                        for r in s.run(cypher, q=query, k=top_k, domain=domain)]
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
                "n.extraction_method AS extraction_method, n.verification AS verification, "
                # `passage` is the paragraph the node's quote came from, and it is
                # what settles a claim: `evidence` averages 32 characters, which
                # names a fact without stating it. Omitting it here made callers
                # re-query Neo4j for the one field they actually needed.
                "n.passage AS passage",
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
        embed_fn: "Callable[[str], list[float]] | None" = None,
        must: str = "",
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
        #
        # `embed_fn` is what makes that saving real. With `embedding` passed in,
        # the caller has already paid for the encoder before learning whether it
        # was needed; handing retrieval the means to encode lets it decide.
        # Measured on verification queries: 21.7 ms -> 5.9 ms.
        lazy = embedding is None and embed_fn is not None
        if fast_path and "fulltext" in channels:
            # Require every term. That a single node contains all of them is
            # real evidence the corpus speaks to the question; a BM25 *score*
            # is not, and trusting one here reintroduced the exact failure the
            # refusal calibration documents — in a small graph a rare word
            # scores high precisely BECAUSE it is rare, so an out-of-corpus
            # query ("kubernetes ingress controller") scored above the floor
            # and was answered instead of refused.
            # Require every term the CORPUS ACTUALLY KNOWS, and drop the rest.
            # Lexical support means one passage literally contains most of what
            # was asked — not that some word scored high. In a small graph a
            # rare word scores high BECAUSE it is rare, which is how "employee
            # stock option vesting schedule" matched "stock" (the broth) and was
            # answered instead of refused.
            probe, covered = self._lexical_evidence(query_text, domain, top_k, must)
            if covered < MIN_TERM_COVERAGE:
                probe = []
            if probe:
                channel_counts["term_coverage"] = round(covered, 2)
                best_fulltext = max(s for _, s in probe)
                if True:
                    ranked["fulltext:0"] = [i for i, _ in probe]
                    channel_counts["fulltext"] = len(probe)
                    channel_counts["fast_path"] = 1
                    embedding = None
                    channels = tuple(c for c in channels if c != "vector")
                    variants = variants[:1]

        if "vector" in channels and embedding is None and lazy:
            embedding = embed_fn(query_text)      # only now is it actually needed
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

        # When one passage already contains nearly the whole query, its
        # neighbours add nothing but a round trip and more nodes to rank. Skip
        # the hop; anything below that bar still gets the full expansion.
        if channel_counts.get("term_coverage", 0.0) >= SETTLED_COVERAGE:
            channels = tuple(c for c in channels if c != "graph")

        if "graph" in channels and hops:
            for row in self.expand_graph(seed_ids, hops=hops):
                # Expansion contributes below any directly-retrieved seed.
                fused.setdefault(row["id"], 0.0)
                fused[row["id"]] += 0.5 / (RRF_K + seeds) * float(row.get("w") or 1.0)
            channel_counts["graph"] = len(fused) - len(seed_ids)

        # Hydrate only what can reach the window. Every node carries its
        # passage — that is the point of it — so pulling the whole fused set
        # ships kilobytes per node the caller will never look at, and that was
        # the largest single cost left in a query.
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        head = ordered[: max(result_window * HYDRATE_FACTOR, DEFAULT_SEEDS)]
        props = self.hydrate([nid for nid, _ in head])
        hits: list[Hit] = []
        for nid, score in head:
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
