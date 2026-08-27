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
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "REFUSAL_TEXT",
    "STRONG_FULLTEXT",
    "balance_by_method",
    "MIN_METHOD_SHARE",
    "PAGE_LANE",
    "RESULT_WINDOW",
    "evidence_lane_enabled",
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
# Re-measured for BGE-m3 on the two-book corpus, 2026-08-26
# (evidence/T83/refusal-calibration.json). The floor is a property of the
# encoder AND the corpus, and carrying over a number calibrated for a different
# model is how a system stops refusing without anyone noticing — which is why
# the encoder change of T83 could not ship without re-running this:
#
#   in corpus      min 0.766   median 0.820   max 0.893   (n=38)
#   out of corpus  min 0.662   median 0.718   max 0.773   (n=24)
#
#     floor   out-of-corpus refused   in-corpus lost
#     0.65    0/24                    0/38
#     0.70    5/24                    0/38     <- the MiniLM floor: near-blind here
#     0.75    21/24                   0/38     <- chosen: most coverage, no loss
#     0.80    24/24                   7/38
#
# The bands OVERLAP under BGE-m3 (gap -0.008), where under MiniLM they separated
# — the two encoders trade discrimination against cross-language reach, and this
# programme needs the reach (architecture §D3). So no floor both refuses
# everything foreign and keeps everything local, and the vector channel alone
# cannot carry the refusal decision. It does not have to: refusal requires the
# LEXICAL channel to be weak as well, and a foreign question that scores 0.78 on
# a passage still shares none of its terms. The end-to-end refusal behaviour is
# what is measured (Q2's unanswerable pairs), never this number on its own.
#
# BM25 still cannot arbitrate support: an out-of-corpus query scores high on a
# rare word precisely BECAUSE it is rare. The lexical channel establishes
# support only when a single node contains EVERY term of the query.
MIN_VECTOR_SIMILARITY = 0.75

#: Similarity above which the semantic channel decides on its own. It was set
#: where no measured out-of-corpus question reached — max 0.773 on the two-book
#: graph, 0.788 on the twelve-book one — and that is no longer true: recalibrated
#: across both lanes on the sixteen-book graph (evidence/T99), out-of-corpus
#: reaches 0.800 on pilot and 0.796 on canon_library. A larger corpus always
#: holds something vaguely close to anything, so the headroom shrinks as books
#: are added and this number cannot be left to age. It costs nothing today only
#: because LEXICAL_FLOOR_IN_BAND is 0.0, which makes the band support anything
#: the floor lets through; the moment that floor rises, this one has to be
#: re-measured with it. Between this and MIN_VECTOR_SIMILARITY the
#: bands OVERLAP and similarity cannot settle the question — a large corpus
#: always holds something vaguely close to anything. There the LEXICON decides:
#: a real question about the corpus uses the corpus's words, and "kubernetes
#: ingress controller tls" shares none of them however similar its vector
#: looks. Calibrated in evidence/T90/refusal-calibration.json.
STRONG_VECTOR = 0.80

#: How many index candidates to pull per requested hit when a domain filter is
#: in play. The vector index is shared across domains and filtering happens
#: after retrieval, so a plain k returns only the fraction that happens to
#: belong to the asked-for corpus.
#: Coverage required in the overlap band. Ablated on the golden set: at 0.6 —
#: the same bar the fast path uses — Q2 fell from 71.4% to 54.8%, because a
#: question asked in natural language does not repeat the corpus's words. What
#: separates a real question from a foreign one is not how MUCH it overlaps but
#: whether it overlaps AT ALL.
LEXICAL_FLOOR_IN_BAND = 0.0

VECTOR_OVERFETCH = 8
#: Ceiling for that widening: past this the scan costs more than the recall it
#: buys, and a domain with nothing to say should refuse rather than dig.
VECTOR_FETCH_CAP = 2000
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

#: Index-name suffix of the evidence lane — the page layer's own vector and
#: fulltext indexes, built by `tools/build_domain_indexes.py`.
PAGE_LANE = "_pages"

#: The window balance below shares out the *results*; by then the damage is
#: done, because a page absent from the candidates cannot be balanced into the
#: window. Measured on the sixteen-book corpus: of 32 grounding failures, 28
#: were pages that never entered the candidate set at all — 90-100% of it was
#: concept nodes — and restricting retrieval to pages alone recovered 12 of 16
#: sampled. So the share is enforced where candidates are *generated*: each
#: lane queries its own index and RRF fuses the four ranked lists.
#:
#: `ENTERPRIPHY_EVIDENCE_LANE=0` turns it off for ablation. Off is also what a
#: graph without page indexes gets, automatically: the lane is skipped when the
#: index does not exist, never silently redirected to the shared one.
def evidence_lane_enabled() -> bool:
    return os.environ.get("ENTERPRIPHY_EVIDENCE_LANE", "1") != "0"


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

def dedupe_by_page(hits: list["Hit"], window: int = RESULT_WINDOW,
                   keep: int = 1) -> list["Hit"]:
    """At most `keep` hits per source page in the head of the list.

    A window is a budget of places to look, and the same page taken four times
    is one place. Measured on the golden set: a ten-slot window held four nodes
    from page 961 and two from page 101, so six of ten slots said the same
    thing twice while the page that answered sat below the cut. Nothing is
    dropped — the surplus moves down — so a caller reading further still sees
    everything.

    Which of the duplicates survives is not a detail. A page node carries the
    whole page; a concept node from that page carries a window inside it, so
    the page node subsumes its rivals and keeping the other one instead throws
    text away. Measured: keeping the best-scored regardless cost two grounded
    records on the canon benchmark, all of them pages that had been displaced
    by their own concepts.
    """
    if window <= 0 or keep <= 0 or not hits:
        return hits
    counts: dict[tuple[str, str], int] = {}
    at: dict[tuple[str, str], int] = {}
    head: list[Hit] = []
    tail: list[Hit] = []
    for h in hits:
        key = (h.source_file or "", h.source_location or h.node_id)
        if len(head) < window and counts.get(key, 0) < keep:
            at[key] = len(head)
            head.append(h)
            counts[key] = counts.get(key, 0) + 1
        elif (key in at and h.extraction_method == "page"
              and head[at[key]].extraction_method != "page"):
            # Same page, wider text: take the slot, and the narrower node goes
            # down with the rest. The slot keeps its rank — this is a swap
            # inside one page, never a promotion over another page's hit.
            tail.append(head[at[key]])
            head[at[key]] = h
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
    """`text` reduced to its content words — OFF by default, and the reason is
    measured rather than assumed.

    Reducing the BM25 query to content words helps a weak multilingual encoder
    and hurts a strong one: the ablation at equal encoder and equal floor
    (evidence/T83/content-filter-ablation.json) gives

        BM25 query      Q2 reported   cross-language   card check   canon
        content words   69.0 %        42.9 %           25/25        99.6 %
        whole phrase    71.4 %        57.1 %           25/25        99.6 %

    so the whole phrase wins on the metric that moved and costs nothing on the
    two approved benchmarks. What DOES matter in both configurations is that
    interrogatives no longer count as required terms (`_FUNCTION_WORDS`), which
    is a different mechanism and stays on.

    `ENTERPRIPHY_CONTENT_FILTER=1` turns the reduction back on — kept because
    the trade-off is a property of the encoder, and the next encoder change has
    to be able to re-run this ablation instead of inheriting a verdict.
    """
    if os.environ.get("ENTERPRIPHY_CONTENT_FILTER") != "1":
        return text
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
    #: Cache of which per-domain vector indexes exist, so the lookup is one
    #: round trip per process rather than one per query.
    _domain_indexes: dict | None = None

    def _load_index_names(self) -> set[str]:
        if HybridRetriever._domain_indexes is None:
            try:
                with self.loader._session() as s:
                    HybridRetriever._domain_indexes = {
                        r["name"] for r in s.run(
                            "SHOW INDEXES YIELD name WHERE name STARTS WITH "
                            "'entity_' RETURN name")}
            except Exception:
                HybridRetriever._domain_indexes = set()
        return HybridRetriever._domain_indexes

    def _domain_index(self, domain: str | None, lane: str = "") -> str:
        """Vector index for `domain` in `lane`, or "" when the lane has none.

        The empty string is deliberate: a missing page index means the graph
        has no evidence lane, and falling back to the domain-wide index would
        make the lane a duplicate of the one it is supposed to complement.
        """
        names = self._load_index_names()
        if not domain:
            return "entity_embedding"
        name = "entity_embedding_" + re.sub(r"[^A-Za-z0-9_]", "_", domain) + lane
        if name in names:
            return name
        return "" if lane else "entity_embedding"

    def _text_index(self, domain: str | None, lane: str = "") -> str:
        """Per-domain text index when one exists, for the same reason the
        vector one exists: the shared index is cut to its best candidates
        before the domain filter can apply."""
        names = self._load_index_names()
        if not domain:
            return "entity_text"
        name = "entity_text_" + re.sub(r"[^A-Za-z0-9_]", "_", domain) + lane
        if name in names:
            return name
        return "" if lane else "entity_text"

    def _merged_vector_indexes(self, lane: str = "") -> list[str]:
        """Every domain's vector index for one lane.

        A domain-less query fans out over these instead of hitting a shared
        index — there is none, by design: it held a second copy of every vector
        and was what ran the database out of memory.
        """
        names = sorted(n for n in self._load_index_names()
                       if n.startswith("entity_embedding_"))
        return [n for n in names
                if n.endswith(PAGE_LANE) == (lane == PAGE_LANE)]

    def lanes(self, domain: str | None) -> tuple[str, ...]:
        """Which lanes this query can use: always the domain-wide one, plus
        the page lane when the graph has indexes for it and it is enabled."""
        if not evidence_lane_enabled():
            return ("",)
        has_pages = (self._domain_index(domain, PAGE_LANE)
                     or (domain is None and self._merged_vector_indexes(PAGE_LANE)))
        return ("", PAGE_LANE) if has_pages else ("",)

    def vector_search(
        self, embedding: list[float], top_k: int = DEFAULT_TOP_K,
        domain: str | None = None, lane: str = "",
    ) -> list[tuple[str, float]]:
        """Nearest neighbours inside one domain.

        The index holds every domain, and `queryNodes` picks its k nearest
        BEFORE the domain filter can apply, so asking for k and filtering after
        returns whatever survives — often nothing. Measured on the two-corpus
        graph: a third of the calibration queries came back empty and the
        semantic channel went blind without any error. So over-fetch, and widen
        until the domain yields enough or the index is exhausted.
        """
        # One index per domain when the graph has them: a shared index makes
        # every corpus pay for every other. Measured on a 149k-node graph
        # holding two corpora — searching the shared index for one of them cost
        # 435 ms and lost a verification, against 33 ms and none when each had
        # its own. Falls back to the shared index where a per-domain one has
        # not been built.
        index = self._domain_index(domain, lane)
        if not index:
            return []
        cypher = (
            f"CALL db.index.vector.queryNodes('{index}', $k, $v) "
            "YIELD node, score "
            "WHERE ($domain IS NULL OR node.domain = $domain) "
            + self._validity_clause("node")
            + " RETURN node.id AS id, score ORDER BY score DESC LIMIT $want"
        )
        with self.loader._session() as s:
            try:
                if domain is None and index == "entity_embedding":
                    # No shared index: it duplicated every domain's vectors and
                    # was what ran the database out of memory. A domain-less
                    # query therefore asks each domain's index and merges —
                    # two round trips instead of one, and no vector stored twice.
                    known = self._merged_vector_indexes(lane)
                    if known:
                        merged: list[tuple[str, float]] = []
                        for idx_name in known:
                            q = cypher.replace(f"'{index}'", f"'{idx_name}'")
                            merged += [(r["id"], r["score"])
                                       for r in s.run(q, k=top_k, want=top_k,
                                                      v=embedding, domain=None)]
                        merged.sort(key=lambda kv: -kv[1])
                        return merged[:top_k]
                if domain is None or index != "entity_embedding":
                    return [(r["id"], r["score"])
                            for r in s.run(cypher, k=top_k, want=top_k, v=embedding,
                                           domain=domain)]
                fetch = top_k * VECTOR_OVERFETCH
                for _ in range(3):
                    rows = [(r["id"], r["score"])
                            for r in s.run(cypher, k=fetch, want=top_k, v=embedding,
                                           domain=domain)]
                    if len(rows) >= top_k or fetch >= VECTOR_FETCH_CAP:
                        return rows
                    fetch = min(fetch * 4, VECTOR_FETCH_CAP)
                return rows
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
                          must: str = "",
                          lane: str = "") -> tuple[list[tuple[str, float]], float]:
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
        index = self._text_index(domain, lane)
        if not index:
            return [], 0.0
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
            f"CALL db.index.fulltext.queryNodes('{index}', $q) "
            "YIELD node, score "
            # Where the cut sits relative to the domain filter decides both
            # recall and cost. On a SHARED index the filter must come first, or
            # the best `probe` candidates are drawn from every corpus and one
            # corpus is left with a handful. On a PER-DOMAIN index the filter is
            # redundant, so the cut goes first and the expensive CONTAINS scan
            # runs over `probe` rows instead of every row the corpus matched —
            # measured, 645 ms against 16 ms per verification.
            + ("WITH node, score ORDER BY score DESC LIMIT $probe WHERE true "
               if index != "entity_text" else
               "WHERE ($domain IS NULL OR node.domain = $domain) "
               "WITH node, score ORDER BY score DESC LIMIT $probe WHERE true ")
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
        require_all: bool = False, require_terms: int = 0, lane: str = "",
    ) -> list[tuple[str, float]]:
        index = self._text_index(domain, lane)
        if not index:
            return []
        cypher = (
            f"CALL db.index.fulltext.queryNodes('{index}', $q) "
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
    def expand_graph(self, seed_ids: list[str], hops: int = 1, limit: int = 40,
                     domain: str | None = None) -> list[dict]:
        """1–2 hop typed expansion, weighted by edge confidence.

        Two things the first version got wrong, both measured on the sixteen-book
        graph:

        * **Identity.** Seeds were matched on `id` alone. Ids are namespaced per
          (book, slice) and the library holds the pilot's own two books, so the
          same id exists in two domains and the walk started from — and ended
          on — nodes belonging to a corpus nobody asked about. Same failure as
          the loader's, same fix: match on (id, domain).
        * **Cost.** 639 ms per expansion — the whole of the p95, on a query
          whose answer was already found. `WHERE ($domain IS NULL OR
          s.domain = $domain)` is idiomatic everywhere else in this file, and
          harmless there because an index call drives the plan; here it IS the
          plan, and the disjunction makes it unusable: profiled, Neo4j fell
          back to NodeByLabelScan over all 199,655 entities and expanded
          426,945 relationships to return 40 neighbours. The predicate is
          therefore built into the statement rather than passed as a parameter
          — one hop, written as one hop, seeking on (id, domain).
        """
        if not seed_ids or hops < 1:
            return []
        hops = min(hops, 2)
        pattern = "-[r]-" if hops == 1 else f"-[r*1..{hops}]-"
        weight = ("coalesce(r.weight, 1.0)" if hops == 1
                  else "reduce(w = 1.0, x IN r | w * coalesce(x.weight, 1.0))")
        scope_s = " AND s.domain = $domain" if domain else ""
        scope_n = " AND n.domain = $domain" if domain else ""
        cypher = (
            f"MATCH (s:Entity) WHERE s.id IN $ids{scope_s} "
            f"MATCH (s){pattern}(n:Entity) "
            f"WHERE NOT n.id IN $ids{scope_n} "
            + self._validity_clause("n")
            + f" WITH n, {weight} AS w "
            "RETURN DISTINCT n.id AS id, w ORDER BY w DESC LIMIT $limit"
        )
        with self.loader._session() as s:
            return [dict(r) for r in s.run(cypher, ids=seed_ids, limit=limit,
                                           domain=domain)]

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
        #: Fraction of the query's content terms found together in one passage.
        #: Computed by the fast-path probe; also the tie-breaker in the overlap
        #: band below, so it is initialised even when that probe is skipped.
        lexical_coverage = 0.0

        # Fast path: probe the lexical channel first. If the corpus literally
        # contains the phrase, the answer is settled and the embedding — the
        # single most expensive step — is never computed.
        #
        # `embed_fn` is what makes that saving real. With `embedding` passed in,
        # the caller has already paid for the encoder before learning whether it
        # was needed; handing retrieval the means to encode lets it decide.
        # Measured on verification queries: 21.7 ms -> 5.9 ms.
        # Which lanes this graph offers. Two lanes mean two ranked lists per
        # channel instead of one, fused by the same RRF: the page layer stops
        # competing with ten times its number of concept nodes for the same
        # candidate slots, and starts competing only with itself.
        lanes = self.lanes(domain)
        page_lane = PAGE_LANE in lanes
        if page_lane:
            channel_counts["evidence_lane"] = 1

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
            lexical_coverage = covered
            if covered < MIN_TERM_COVERAGE:
                probe = []
            if probe:
                # The evidence lane probes the page index too, and here it is
                # the whole point: the corpus literally contains the phrase, and
                # what settles the claim is the PAGE carrying it — which the
                # domain-wide probe rarely returns, because concept nodes fill
                # it.
                #
                # What the page probe must NOT do is decide that this path was
                # taken. Letting it trigger the fast path on its own cost 14
                # points of cross-language page recall (64.3% -> 50.0%,
                # evidence/T99/context-recall): the fast path skips the
                # encoder, and a question asked in French about an English book
                # shares no words with the page that answers it — the semantic
                # channel is the only way across. A French page containing the
                # French words is not evidence that the encoder is unnecessary.
                page_probe: list[tuple[str, float]] = []
                if page_lane:
                    page_probe, page_covered = self._lexical_evidence(
                        query_text, domain, top_k, must, lane=PAGE_LANE)
                    if page_covered >= MIN_TERM_COVERAGE and page_probe:
                        lexical_coverage = max(lexical_coverage, page_covered)
                        channel_counts["term_coverage_pages"] = round(page_covered, 2)
                        ranked["fulltext:pages"] = [i for i, _ in page_probe]
                        channel_counts["fulltext_pages"] = len(page_probe)
                channel_counts["term_coverage"] = round(lexical_coverage, 2)
                best_fulltext = max(s for _, s in probe + page_probe)
                ranked["fulltext:0"] = [i for i, _ in probe]
                channel_counts["fulltext"] = len(probe)
                channel_counts["fast_path"] = 1
                variants = variants[:1]
                # The fast path settles SUPPORT — the corpus contains the
                # phrase — and used to also skip the encoder, which is the most
                # expensive step. That was measured on a two-book corpus, where
                # full term coverage meant one passage. On sixteen books it
                # means dozens: coverage 1.0 now says the words are present, not
                # that the lexical order discriminates. Measured, all five
                # remaining grounding failures took this path, and the page that
                # answered sat at vector rank 1 in a channel that never ran.
                #
                # The optimisation was worth it when a query cost 400 ms against
                # an 800 ms gate. It costs 33 ms now. So support still comes
                # from the lexicon, and the encoder runs anyway.
                # Measured both ways on the sixteen-book graph (evidence/T99):
                #
                #   encoder skipped (default)   canon 95.5%   Q2 page recall 78.9%
                #   encoder kept                canon 87.0%   Q2 page recall 86.8%
                #
                # Neither dominates, because the two benchmarks ask different
                # things: canon looks for a passage the caller can already
                # quote, which is a lexical problem, and the semantic candidates
                # crowd it out; Q2 asks questions in natural language, which is
                # the opposite. So this is an operating point, not a bug, and it
                # is exposed rather than decided in secret:
                # ENTERPRIPHY_FAST_PATH_KEEPS_VECTOR=1 trades eight points of
                # documentary grounding for eight points of question recall.
                if os.environ.get("ENTERPRIPHY_FAST_PATH_KEEPS_VECTOR") != "1":
                    embedding = None
                    channels = tuple(c for c in channels if c != "vector")

        if "vector" in channels and embedding is None and lazy:
            embedding = embed_fn(query_text)      # only now is it actually needed
        if "vector" in channels and embedding is not None:
            vec = self.vector_search(embedding, top_k=top_k, domain=domain)
            ranked["vector"] = [i for i, _ in vec]
            channel_counts["vector"] = len(vec)
            best_vector = max((s for _, s in vec), default=0.0)
            if page_lane:
                pvec = self.vector_search(embedding, top_k=top_k, domain=domain,
                                          lane=PAGE_LANE)
                if pvec:
                    ranked["vector:pages"] = [i for i, _ in pvec]
                    channel_counts["vector_pages"] = len(pvec)
                    best_vector = max(best_vector, max(s for _, s in pvec))

        if "fulltext" in channels and not channel_counts.get("fast_path"):
            for idx, variant in enumerate(variants):
                ft = self.fulltext_search(variant, top_k=top_k, domain=domain)
                if ft:
                    ranked[f"fulltext:{idx}"] = [i for i, _ in ft]
                    best_fulltext = max(best_fulltext, max(s for _, s in ft))
                if page_lane:
                    ftp = self.fulltext_search(variant, top_k=top_k, domain=domain,
                                               lane=PAGE_LANE)
                    if ftp:
                        ranked[f"fulltext:pages:{idx}"] = [i for i, _ in ftp]
                        best_fulltext = max(best_fulltext, max(s for _, s in ftp))
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
            if best_vector >= STRONG_VECTOR:
                supported = True
            elif best_vector < min_vector_similarity:
                supported = False
            else:
                # The overlap band: the vector cannot tell a real question from
                # a foreign one, so require that the corpus contain the words
                # actually asked about. Measured on the twelve-book graph: with
                # similarity alone, 10 of 24 foreign questions were answered.
                supported = lexical_coverage >= LEXICAL_FLOOR_IN_BAND
                channel_counts["decided_by"] = "lexicon"
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
            for row in self.expand_graph(seed_ids, hops=hops, domain=domain):
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
        if os.environ.get("ENTERPRIPHY_PAGE_DEDUPE", "1") != "0":
            hits = dedupe_by_page(hits, window=result_window)
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
