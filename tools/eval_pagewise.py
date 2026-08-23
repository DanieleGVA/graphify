#!/usr/bin/env python3
"""Extraction-neutral evaluation: did retrieval reach the right PAGE?

`golden-qa-v1.json` was generated mechanically from the structural extraction:
queries are distinctive terms taken from structural nodes' text, `answer_span`
is a structural node's evidence quote, and `ground_truth_node` is a structural
node id. Three of the four headline metrics are therefore biased toward that
one extraction, two of them fatally:

  * `node_level_recall`  — other extractions do not have those ids at all, so
                           it is 0 by construction, not by failure.
  * `q2_context_contains_answer` — requires the structural node's own (often
                           OCR-garbled) text to reappear in the retrieved
                           context. A cleaner extraction cannot reproduce it.
  * `q3_recall_at_10`    — the query terms are lifted verbatim from structural
                           excerpts, which the structural graph then matches
                           exactly by fulltext.

Only Q1 faithfulness is neutral, because it checks each served claim against
that claim's own stored source.

This harness adds a metric that privileges neither: every extraction records
where a fact came from — book and page range — independently. So ask whether
any top-10 hit lands on the same book AND overlaps the ground-truth pages.
Reported at exact overlap and with a ±2-page tolerance, since page attribution
at a slice boundary is legitimately fuzzy.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from graphify_ent.embed import Embedder
from graphify_ent.loader import Neo4jLoader
from graphify_ent.retrieval import HybridRetriever

_PAGES = re.compile(r"(\d+)\s*-\s*(\d+)")
TOP_N = 10


def pages_of(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    m = _PAGES.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m = re.search(r"(\d+)", text)
    return (int(m.group(1)), int(m.group(1))) if m else None


def overlaps(a: tuple[int, int], b: tuple[int, int], tol: int = 0) -> bool:
    return a[0] - tol <= b[1] and b[0] - tol <= a[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--label", default="grafo")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    pairs = [p for p in json.loads(args.golden.read_text())["pairs"]
             if p.get("ground_truth_doc") and pages_of(p.get("source_location"))]

    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()

    rows, lat = [], []
    try:
        for p in pairs:
            gt_pages = pages_of(p["source_location"])
            emb = embedder.encode([p["query"]])[0]
            t0 = time.perf_counter()
            res = retriever.query(p["query"], embedding=emb,
                                  channels=("vector", "fulltext", "graph"), hops=1,
                                  domain="pilot")
            lat.append((time.perf_counter() - t0) * 1000)
            hits = res.hits[:TOP_N]
            props = retriever.hydrate([h.node_id for h in hits])
            doc_hit = any(h.source_file == p["ground_truth_doc"] for h in hits)
            page_hit = page_hit_tol = False
            for h in hits:
                if h.source_file != p["ground_truth_doc"]:
                    continue
                hp = pages_of((props.get(h.node_id) or {}).get("source_location"))
                if not hp:
                    continue
                page_hit = page_hit or overlaps(hp, gt_pages)
                page_hit_tol = page_hit_tol or overlaps(hp, gt_pages, tol=2)
            rows.append({"id": p["id"], "kind": p.get("kind"),
                         "book": p["ground_truth_doc"], "doc_hit": doc_hit,
                         "page_hit": page_hit, "page_hit_tol2": page_hit_tol,
                         "hits": len(hits)})
    finally:
        loader.close()

    def pct(key, sel=None):
        s = [r for r in rows if sel is None or r["kind"] == sel]
        return round(100 * sum(1 for r in s if r[key]) / len(s), 2) if s else None

    report = {
        "label": args.label, "queries": len(rows),
        "doc_recall_at_10": pct("doc_hit"),
        "page_recall_at_10": pct("page_hit"),
        "page_recall_at_10_tol2": pct("page_hit_tol2"),
        "page_recall_monolingual": pct("page_hit", "monolingual"),
        "page_recall_cross_language": pct("page_hit", "cross_language"),
        "queries_with_no_hits": sum(1 for r in rows if r["hits"] == 0),
        "per_book": {
            b: {"n": len([r for r in rows if r["book"] == b]),
                "page_recall": round(100 * sum(1 for r in rows if r["book"] == b
                                               and r["page_hit"])
                                     / max(len([r for r in rows if r["book"] == b]), 1), 1)}
            for b in sorted({r["book"] for r in rows})},
        "latency_ms_p95": round(statistics.quantiles(lat, n=20)[-1], 1) if len(lat) > 1 else None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"report": report, "per_query": rows},
                                        indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
