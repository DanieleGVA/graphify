"""Canonical quantity facts (ADR-0004, Q1-B).

Every figure in a stored passage, read once with the programme's one grammar
(`graphify_ent.quantities`) and anchored to the words it belongs to. The
verify_claim defect existed because a page is a bag of numbers with no owners:
"1 gal" was confirmed as a milk quantity because it merely appeared on the
page — it was the recipe's YIELD. A fact makes the ownership explicit and
computes it ONCE, at derivation, instead of re-guessing it per claim with
text heuristics.

Two anchor kinds, and the difference is declared, never hidden:

  table-row     the figure sits in a flattened PDF table; its owner is the
                nearest line above that carries words ("Milk" over "5 qt").
                Structure read directly — the exact anchor.
  prose-window  the figure sits in running text; its owner is what the same
                sentence says around it, within a bounded window. Proximity —
                weaker, and it says so via `uncertainty` (ADR: an exact anchor
                outranks a proximity anchor).

`raw_text` is cut verbatim from the source passage and must satisfy
`retrieval.verify_evidence_binding` — a fact whose quote cannot be found in
its own source is dropped, not shipped. Derivation is pure and deterministic;
`materialize` batch-writes `(:QuantityFact)-[:ANCHORED_TO]->(:Entity)` from
passages ALREADY in the graph — no re-ingest — and is idempotent by fact id.
Until T60 materializes facts everywhere, verify derives them on the fly
through this same function (ADR-0004).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from graphify_ent.quantities import (
    Quantity,
    UnitTable,
    default_table,
    norm_spans,
    scan,
    to_original,
)

__all__ = ["Fact", "TABLE_ROW", "PROSE_WINDOW", "ANCHOR_UNCERTAINTY",
           "derive", "materialize"]

TABLE_ROW = "table-row"
PROSE_WINDOW = "prose-window"

#: Declared attribution uncertainty per anchor kind — same discipline as the
#: resolution classes: a proximity read is not a structural read, and hiding
#: the difference is how approximation turns into false precision.
ANCHOR_UNCERTAINTY = {TABLE_ROW: 0.0, PROSE_WINDOW: 0.15}

#: Half-width, in normalised characters, of the prose window around a figure.
#: Bounded AND clipped at sentence ends: 90 unclipped characters were measured
#: pulling a yield figure into the window of the milk that a previous sentence
#: mentioned (evidence/verify-canonical-prototype/).
WINDOW = 60

#: A line longer than this inside a tabular page is a flattened prose tail
#: ("[segue da p. 380] ..."), not a cell — it is read in prose mode.
FLAT_LINE = 100

#: Tabular detection, same measured rule as the verifier's: a table in
#: extracted PDF text is lines whose whole content IS figures.
MIN_FIGURES, MIN_CELLS = 6, 3

_SEPS = re.compile(r"[\s/,;:·|–—()\[\]-]")


@dataclass(frozen=True)
class Fact:
    """One owned figure. `span` locates `raw_text` in the ORIGINAL passage."""

    value_lo: float
    value_hi: float
    unit_base: str
    raw_text: str
    anchor_text: str
    anchor_kind: str
    uncertainty: float
    span: tuple[int, int]

    def fact_id(self, node_id: str, grammar_version: str) -> str:
        """Deterministic identity: same node, same figure, same grammar →
        same fact. This is what makes the batch idempotent."""
        key = "|".join((node_id, grammar_version, self.unit_base,
                        f"{self.value_lo:g}", f"{self.value_hi:g}",
                        f"{self.span[0]}-{self.span[1]}"))
        return hashlib.sha1(key.encode()).hexdigest()

    def as_props(self) -> dict:
        return {"value_lo": self.value_lo, "value_hi": self.value_hi,
                "unit_base": self.unit_base, "raw_text": self.raw_text,
                "anchor_text": self.anchor_text, "anchor_kind": self.anchor_kind,
                "uncertainty": self.uncertainty}


def _residual(line: str, table: UnitTable) -> str:
    """What is left of a line once its figures are removed — the words."""
    body, _ = norm_spans(line)
    cut = body
    for q in scan(line, table):
        s, e = q.span
        cut = cut[:s] + " " * (e - s) + cut[e:]
    return _SEPS.sub("", cut)


def _is_tabular(text: str, table: UnitTable) -> bool:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    figures = cells = 0
    for ln in lines:
        if len(ln) > FLAT_LINE:
            continue
        n = len(scan(ln, table))
        figures += n
        if n and not _residual(ln, table):
            cells += 1
    return figures >= MIN_FIGURES and cells >= MIN_CELLS


def _prose(text: str, table: UnitTable, origin: int = 0) -> list[Fact]:
    """Figures in running text, each owned by its sentence-clipped window."""
    body, idx = norm_spans(text)
    bounds = [0] + [m.end() for m in re.finditer(r"[.!?](?=\s)", body)] + [len(body)]
    out: list[Fact] = []
    for q in scan(text, table):
        s, e = q.span
        lo = max(b for b in bounds if b <= s)
        hi = min(b for b in bounds if b >= e)
        ws, we = to_original((max(lo, s - WINDOW), min(hi, e + WINDOW)), idx)
        raw = text[ws:we]
        os_, oe = to_original(q.span, idx)
        out.append(Fact(q.lo, q.hi, q.unit, raw, raw, PROSE_WINDOW,
                        ANCHOR_UNCERTAINTY[PROSE_WINDOW],
                        (origin + os_, origin + oe)))
    return out


def _line_offsets(text: str) -> list[tuple[int, str]]:
    at, out = 0, []
    for ln in text.splitlines(keepends=True):
        stripped = ln.rstrip("\r\n")
        if stripped.strip():
            out.append((at, stripped))
        at += len(ln)
    return out


def derive(passage: str, table: UnitTable | None = None) -> list[Fact]:
    """Every owned figure in one passage, evidence-bound.

    A fact whose `raw_text` fails the binding check against the passage is
    dropped — by construction it should never happen, and a quote that cannot
    be found in its own source must not exist (blueprint §3, Q1).
    """
    t = table or default_table()
    if not (passage or "").strip():
        return []
    facts: list[Fact] = []
    if _is_tabular(passage, t):
        lines = _line_offsets(passage)
        resid = [_residual(ln, t) for _, ln in lines]
        for i, (at, ln) in enumerate(lines):
            if len(ln) > FLAT_LINE:
                facts.extend(_prose(ln, t, origin=at))
                continue
            qs = scan(ln, t)
            if not qs:
                continue
            # The owner is the nearest line ABOVE that carries words — the
            # same one-to-one rule R1 bought with the COFFEE SYRUP failure.
            j = i if resid[i] else next(
                (k for k in range(i - 1, -1, -1) if resid[k]), None)
            if j is None:
                continue
            anchor = lines[j][1] if j != i else ln
            raw = "\n".join(l for _, l in lines[j: i + 1])
            _, idx = norm_spans(ln)
            for q in qs:
                os_, oe = to_original(q.span, idx)
                facts.append(Fact(q.lo, q.hi, q.unit, raw, anchor, TABLE_ROW,
                                  ANCHOR_UNCERTAINTY[TABLE_ROW],
                                  (at + os_, at + oe)))
    else:
        facts = _prose(passage, t)

    from graphify_ent.retrieval import verify_evidence_binding

    return [f for f in facts if verify_evidence_binding(f.raw_text, passage)]


# ---------------------------------------------------------------- batch
_CONSTRAINT = ("CREATE CONSTRAINT quantity_fact_id IF NOT EXISTS "
               "FOR (f:QuantityFact) REQUIRE f.fact_id IS UNIQUE")

_WRITE = """
UNWIND $rows AS r
// id alone is NOT unique across domains: the same book loaded in two
// domains shares content-derived ids, and matching by id anchored 64k
// facts to the other domain's twin nodes (measured, first live run).
MATCH (n:Entity {id: r.node_id, domain: $domain})
MERGE (f:QuantityFact {fact_id: r.fact_id})
ON CREATE SET f += r.props,
              f.domain = $domain,
              f.grammar_version = $gv,
              f.ingested_at = $now,
              f.valid_from = n.valid_from,
              f.valid_to = n.valid_to,
              f.invalidated_at = null,
              f.created = true
MERGE (f)-[:ANCHORED_TO]->(n)
WITH f, f.created AS created
REMOVE f.created
RETURN count(CASE WHEN created THEN 1 END) AS made
"""

_NODES = """
MATCH (n:Entity {domain: $domain})
WHERE n.passage IS NOT NULL AND n.invalidated_at IS NULL
RETURN n.id AS id, n.passage AS passage
"""


def materialize(loader, domain: str, table: UnitTable | None = None,
                batch: int = 200) -> dict:
    """Derive facts for every live passage of a domain and write them.

    Idempotent: re-running MERGEs onto the same fact ids. Nothing is deleted;
    facts of an invalidated source are handled by the temporal engine like any
    other assertion-bearing node.
    """
    t = table or default_table()
    gv = f"units-v{t.version}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats = {"nodes": 0, "facts": 0, "created": 0, "grammar_version": gv}
    with loader._session() as s:
        s.run(_CONSTRAINT)
        rows: list[dict] = []

        def flush():
            if not rows:
                return 0
            # list(rows): the driver may hold the parameter object past this
            # call, and clear() below would empty it under its feet.
            made = s.run(_WRITE, rows=list(rows), domain=domain, gv=gv,
                         now=now).single()["made"]
            rows.clear()
            return made

        for rec in s.run(_NODES, domain=domain):
            stats["nodes"] += 1
            for f in derive(rec["passage"], t):
                stats["facts"] += 1
                rows.append({"node_id": rec["id"],
                             "fact_id": f.fact_id(rec["id"], gv),
                             "props": f.as_props()})
                if len(rows) >= batch:
                    stats["created"] += flush()
        stats["created"] += flush()
    return stats


def main() -> None:
    import argparse
    import json

    from graphify_ent.loader import Neo4jLoader

    ap = argparse.ArgumentParser(
        description="Batch-derive :QuantityFact from stored passages (no re-ingest)")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()
    with Neo4jLoader() as loader:
        print(json.dumps(materialize(loader, args.domain, batch=args.batch)))


if __name__ == "__main__":
    main()
