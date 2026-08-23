"""Phase 6.3–6.5 — clash blocking, adjudication, resolution.

Architecture §2.3. Three stages, deliberately separated so the expensive one
runs on the smallest possible candidate set:

6.3 **Blocking** — candidate pairs only where cheap signals already agree:
    same dedup/concept cluster, OR embedding cosine > 0.85 within one domain,
    and both nodes currently valid. Checkpoint: pair count must be ≤ 5× node
    count, else thresholds get tuned *before* any adjudication spend.

6.4 **Adjudication** — batched LLM verdicts
    `SAME | COMPLEMENTARY | CONTRADICTORY | SUPERSEDES` with a rationale that
    must quote both source excerpts (same evidence-binding discipline as
    `graphify/llm.py:_bind_node_evidence`). Verdicts are cached by
    pair-content-hash, so unchanged pairs are never re-adjudicated.

6.5 **Resolution** — policy chain authority → recency → confidence.
    **Zero silent winners**: an auto-resolution always writes its rationale and
    the policy step that decided it onto the `[:SUPERSEDED_BY]` edge; anything
    unresolved becomes `[:CONTRADICTS]` plus a human review queue entry.

The adjudicator is an injected callable, so the deterministic pipeline is fully
testable without credentials and a real LLM backend drops in unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "VERDICTS",
    "Adjudicator",
    "ClashEngine",
    "ClashPair",
    "Resolution",
    "Verdict",
    "VerdictCache",
    "pair_content_hash",
    "resolve",
]

VERDICTS = ("SAME", "COMPLEMENTARY", "CONTRADICTORY", "SUPERSEDES")
DEFAULT_COSINE = 0.85
PAIR_BUDGET_MULTIPLIER = 5  # checkpoint: pairs ≤ 5× nodes


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ClashPair:
    a_id: str
    b_id: str
    a_text: str = ""
    b_text: str = ""
    similarity: float = 0.0
    reason: str = "embedding"

    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.a_id, self.b_id)))  # type: ignore[return-value]


@dataclass
class Verdict:
    verdict: str
    rationale: str = ""
    confidence: float = 0.0
    a_excerpt: str = ""
    b_excerpt: str = ""
    winner: str | None = None

    def is_valid(self) -> bool:
        return self.verdict in VERDICTS


@dataclass
class Resolution:
    pair: ClashPair
    verdict: Verdict
    action: str                      # "supersede" | "contradict" | "none"
    winner: str | None = None
    loser: str | None = None
    policy: str | None = None        # which chain step decided it
    rationale: str = ""

    def as_dict(self) -> dict:
        return {
            "a": self.pair.a_id, "b": self.pair.b_id,
            "verdict": self.verdict.verdict, "action": self.action,
            "winner": self.winner, "loser": self.loser,
            "policy": self.policy, "rationale": self.rationale,
        }


def pair_content_hash(pair: ClashPair) -> str:
    """Cache key over *content*, so unchanged pairs are never re-adjudicated."""
    a, b = sorted([(pair.a_id, pair.a_text), (pair.b_id, pair.b_text)])
    payload = f"{a[0]}|{a[1][:1000]}|{b[0]}|{b[1][:1000]}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


class VerdictCache:
    """Disk-backed verdict cache (never re-adjudicate unchanged pairs)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else Path(".graphify-ent/verdicts.json")
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Verdict | None:
        raw = self._data.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return Verdict(**raw)

    def put(self, key: str, verdict: Verdict) -> None:
        self._data[key] = {
            "verdict": verdict.verdict, "rationale": verdict.rationale,
            "confidence": verdict.confidence, "a_excerpt": verdict.a_excerpt,
            "b_excerpt": verdict.b_excerpt, "winner": verdict.winner,
        }

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))


#: An adjudicator maps a batch of pairs to verdicts, in order.
Adjudicator = Callable[[Sequence[ClashPair]], list[Verdict]]


def resolve(
    pair: ClashPair,
    verdict: Verdict,
    props: dict[str, dict],
) -> Resolution:
    """Apply the resolution chain: authority → recency → confidence.

    Returns a Resolution that always records *which* step decided, so no
    auto-resolution is ever silent (architecture §2.3, rule d).
    """
    if verdict.verdict in ("SAME", "COMPLEMENTARY"):
        return Resolution(pair, verdict, action="none",
                          rationale="not a conflict: " + verdict.verdict.lower())

    a, b = props.get(pair.a_id, {}), props.get(pair.b_id, {})

    if verdict.verdict == "SUPERSEDES" and verdict.winner in (pair.a_id, pair.b_id):
        winner = verdict.winner
        loser = pair.b_id if winner == pair.a_id else pair.a_id
        return Resolution(pair, verdict, action="supersede", winner=winner, loser=loser,
                          policy="adjudicated-supersedes", rationale=verdict.rationale)

    # a. authority (lower source_rank wins)
    ra, rb = a.get("source_rank"), b.get("source_rank")
    if isinstance(ra, int) and isinstance(rb, int) and ra != rb:
        winner, loser = (pair.a_id, pair.b_id) if ra < rb else (pair.b_id, pair.a_id)
        return Resolution(pair, verdict, action="supersede", winner=winner, loser=loser,
                          policy="authority",
                          rationale=f"source_rank {min(ra, rb)} outranks {max(ra, rb)}")

    # b. recency (needs a confident date on both sides)
    da, db = a.get("doc_date"), b.get("doc_date")
    ca = float(a.get("doc_date_confidence") or 0)
    cb = float(b.get("doc_date_confidence") or 0)
    if da and db and da != db and min(ca, cb) >= 0.8:
        winner, loser = (pair.a_id, pair.b_id) if da > db else (pair.b_id, pair.a_id)
        return Resolution(pair, verdict, action="supersede", winner=winner, loser=loser,
                          policy="recency",
                          rationale=f"doc_date {max(da, db)} is newer than {min(da, db)}")

    # c. extraction confidence (EXTRACTED > INFERRED > AMBIGUOUS; OCR down-ranked)
    order = {"EXTRACTED": 3, "INFERRED": 2, "AMBIGUOUS": 1}
    sa = order.get(a.get("confidence", "EXTRACTED"), 2) - (
        1 if a.get("extraction_method") == "ocr" else 0)
    sb = order.get(b.get("confidence", "EXTRACTED"), 2) - (
        1 if b.get("extraction_method") == "ocr" else 0)
    if sa != sb:
        winner, loser = (pair.a_id, pair.b_id) if sa > sb else (pair.b_id, pair.a_id)
        return Resolution(pair, verdict, action="supersede", winner=winner, loser=loser,
                          policy="confidence",
                          rationale="higher extraction confidence / non-OCR provenance")

    # d. no silent winners — both stay valid, flagged, and queued for a human
    return Resolution(pair, verdict, action="contradict", policy=None,
                      rationale="policy chain could not resolve; both remain valid")


class ClashEngine:
    def __init__(self, loader, cache: VerdictCache | None = None):
        self.loader = loader
        self.cache = cache or VerdictCache()

    # -- 6.3 blocking -------------------------------------------------------
    def node_count(self, domain: str | None = None) -> int:
        with self.loader._session() as s:
            return s.run(
                "MATCH (n:Entity) WHERE ($d IS NULL OR n.domain = $d) "
                "AND n.invalidated_at IS NULL RETURN count(n) AS c", d=domain
            ).single()["c"]

    def block_pairs(
        self,
        domain: str | None = None,
        cosine: float = DEFAULT_COSINE,
        limit: int = 20_000,
    ) -> list[ClashPair]:
        """Candidate pairs: high cosine within a domain, both currently valid.

        Uses the vector index per node rather than an O(n²) cross join.
        """
        pairs: dict[tuple[str, str], ClashPair] = {}
        with self.loader._session() as s:
            rows = s.run(
                "MATCH (n:Entity) WHERE n.embedding IS NOT NULL "
                "AND n.invalidated_at IS NULL "
                "AND ($d IS NULL OR n.domain = $d) "
                "RETURN n.id AS id, n.embedding AS emb, n.text_excerpt AS text "
                "LIMIT $limit",
                d=domain, limit=limit,
            )
            nodes = [dict(r) for r in rows]

            for node in nodes:
                res = s.run(
                    "CALL db.index.vector.queryNodes('entity_embedding', 6, $v) "
                    "YIELD node, score WHERE node.id <> $id AND score >= $cos "
                    "AND node.invalidated_at IS NULL "
                    "AND ($d IS NULL OR node.domain = $d) "
                    "RETURN node.id AS id, node.text_excerpt AS text, score",
                    v=node["emb"], id=node["id"], cos=cosine, d=domain,
                )
                for r in res:
                    pair = ClashPair(
                        a_id=node["id"], b_id=r["id"],
                        a_text=node.get("text") or "", b_text=r.get("text") or "",
                        similarity=float(r["score"]), reason="embedding",
                    )
                    pairs.setdefault(pair.key(), pair)
        return list(pairs.values())

    def blocking_checkpoint(self, pairs: list[ClashPair], domain: str | None = None) -> dict:
        """Plan §6.3 gate: pair count must be ≤ 5× node count before 6.4 spend."""
        n = self.node_count(domain)
        budget = n * PAIR_BUDGET_MULTIPLIER
        return {
            "nodes": n, "pairs": len(pairs), "budget": budget,
            "within_budget": len(pairs) <= budget,
            "ratio": round(len(pairs) / n, 3) if n else None,
        }

    # -- 6.4 adjudication ---------------------------------------------------
    def adjudicate(
        self, pairs: list[ClashPair], adjudicator: Adjudicator, batch_size: int = 20
    ) -> list[tuple[ClashPair, Verdict]]:
        """Batched adjudication with a content-hash verdict cache."""
        out: list[tuple[ClashPair, Verdict]] = []
        pending: list[ClashPair] = []

        def flush_pending():
            if not pending:
                return
            verdicts = adjudicator(pending)
            for p, v in zip(pending, verdicts):
                if not v.is_valid():
                    v = Verdict(verdict="COMPLEMENTARY",
                                rationale="adjudicator returned an invalid verdict")
                # Evidence binding: the rationale must quote both sources.
                if v.verdict == "CONTRADICTORY" and not (v.a_excerpt and v.b_excerpt):
                    v.confidence = min(v.confidence, 0.5)
                self.cache.put(pair_content_hash(p), v)
                out.append((p, v))
            pending.clear()

        for pair in pairs:
            cached = self.cache.get(pair_content_hash(pair))
            if cached is not None:
                out.append((pair, cached))
                continue
            pending.append(pair)
            if len(pending) >= batch_size:
                flush_pending()
        flush_pending()
        self.cache.flush()
        return out

    # -- 6.5 resolution -----------------------------------------------------
    def hydrate_props(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        with self.loader._session() as s:
            rows = s.run(
                "MATCH (n:Entity) WHERE n.id IN $ids "
                "RETURN n.id AS id, n.source_rank AS source_rank, n.doc_date AS doc_date, "
                "n.doc_date_confidence AS doc_date_confidence, n.confidence AS confidence, "
                "n.extraction_method AS extraction_method",
                ids=ids,
            )
            return {r["id"]: dict(r) for r in rows}

    def apply_resolutions(
        self, judged: list[tuple[ClashPair, Verdict]], dry_run: bool = True
    ) -> list[Resolution]:
        """Write SUPERSEDED_BY / CONTRADICTS with a full policy trail.

        `dry_run` defaults to True: auto-resolution stays disabled until the G4
        human signature (CLAUDE.md HUMAN GATES rule).
        """
        ids = sorted({i for p, _ in judged for i in (p.a_id, p.b_id)})
        props = self.hydrate_props(ids)
        resolutions = [resolve(p, v, props) for p, v in judged]

        if dry_run:
            return resolutions

        supersedes = [r for r in resolutions if r.action == "supersede"]
        contradicts = [r for r in resolutions if r.action == "contradict"]

        with self.loader._session() as s:
            if supersedes:
                s.execute_write(lambda tx: tx.run(
                    "UNWIND $rows AS row "
                    "MATCH (w:Entity {id: row.winner}) MATCH (l:Entity {id: row.loser}) "
                    "MERGE (l)-[r:SUPERSEDED_BY]->(w) "
                    "SET r.detected_at = $at, r.policy = row.policy, "
                    "    r.rationale = row.rationale, r.resolved_by = 'auto' "
                    "SET l.valid_to = coalesce(l.valid_to, $at) "
                    "RETURN count(r)",
                    rows=[{"winner": r.winner, "loser": r.loser, "policy": r.policy,
                           "rationale": r.rationale} for r in supersedes],
                    at=_now(),
                ).consume())
            if contradicts:
                s.execute_write(lambda tx: tx.run(
                    "UNWIND $rows AS row "
                    "MATCH (a:Entity {id: row.a}) MATCH (b:Entity {id: row.b}) "
                    "MERGE (a)-[r:CONTRADICTS]-(b) "
                    "SET r.detected_at = $at, r.rationale = row.rationale, "
                    "    r.adjudication_confidence = row.confidence, r.review_status = 'open' "
                    "RETURN count(r)",
                    rows=[{"a": r.pair.a_id, "b": r.pair.b_id, "rationale": r.rationale,
                           "confidence": r.verdict.confidence} for r in contradicts],
                    at=_now(),
                ).consume())
        return resolutions

    def conflict_report(self, resolutions: list[Resolution], run_id: str) -> str:
        """Human review queue (execution plan 6.5)."""
        lines = [f"# Conflict report — run {run_id}", "",
                 f"Generated {_now()}. {len(resolutions)} adjudicated pairs.", ""]
        open_items = [r for r in resolutions if r.action == "contradict"]
        auto = [r for r in resolutions if r.action == "supersede"]

        lines += [f"## Open contradictions requiring a human ({len(open_items)})", ""]
        for r in open_items:
            lines += [
                f"- **{r.pair.a_id}** vs **{r.pair.b_id}** (cosine {r.pair.similarity:.3f})",
                f"  - rationale: {r.rationale}",
                f"  - resolve with: `review_cli.py resolve {r.pair.a_id}:{r.pair.b_id} "
                f"--winner <node_id>`",
            ]
        lines += ["", f"## Auto-resolved with policy trail ({len(auto)})", ""]
        for r in auto:
            lines.append(f"- {r.loser} superseded by {r.winner} — policy `{r.policy}`: "
                         f"{r.rationale}")
        return "\n".join(lines)
