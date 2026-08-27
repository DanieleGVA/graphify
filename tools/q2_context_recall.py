#!/usr/bin/env python3
"""Does the context handed to the answerer contain the page that answers?

Q2 measures a chain — retrieval, then an LLM answer, then an LLM judge — and
the last two links move on their own: the same golden set, same graph and same
code scored 54.8%, 61.9%, 64.3%, 66.7% and 66.7% across the T90 series. A
change to retrieval cannot be judged on a number with that much travel in it.

This measures the first link only, and it is deterministic: for every
answerable pair, does the expected page appear among the top-k hits the
answerer would have been shown? No model is called, so the same command run
twice gives the same figure, and an ablation means something.

    python tools/q2_context_recall.py --domain pilot --json ../evidence/T99/context-recall.json
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


def pages_of(loc: str | None) -> set[int]:
    if not loc:
        return set()
    m = re.search(r"(\d+)\s*-\s*(\d+)", loc)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return set(range(min(a, b), max(a, b) + 1))
    m = re.search(r"(\d+)", loc)
    return {int(m.group(1))} if m else set()


def book_matches(expected: str, source_file: str) -> bool:
    """The golden set names a book the way a person would ("Professional Chef");
    the graph holds the file name. Compare on the words they share."""
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", expected or "")]
    low = (source_file or "").lower()
    return bool(words) and all(w.lower() in low for w in words)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("../eval/q2-golden-v1.json"))
    ap.add_argument("--domain", default="pilot")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", type=Path,
                    default=Path("../evidence/T99/context-recall.json"))
    args = ap.parse_args()

    pairs = [p for p in json.loads(args.golden.read_text())["pairs"]
             if p["kind"] != "unanswerable"]
    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    embedder.encode(["warm up"])
    enc = lambda q: embedder.encode([q])[0]  # noqa: E731

    rows = []
    try:
        for p in pairs:
            t0 = time.perf_counter()
            res = retriever.query(p["query"], embed_fn=enc, domain=args.domain,
                                  channels=("vector", "fulltext", "graph"), hops=1)
            ms = (time.perf_counter() - t0) * 1000
            ids = [h.node_id for h in res.hits[: args.top]]
            props = retriever.hydrate(ids) if ids else {}
            want_page = int(p["page"]) if str(p.get("page", "")).isdigit() else None
            hit_book = hit_page = False
            for nid in ids:
                d = props.get(nid) or {}
                if not book_matches(p.get("book", ""), d.get("source_file", "")):
                    continue
                hit_book = True
                if want_page is None or want_page in pages_of(d.get("source_location")):
                    hit_page = True
                    break
            rows.append({"id": p.get("id"), "kind": p["kind"], "lang": p.get("lang"),
                         "query": p["query"], "book": p.get("book"), "page": p.get("page"),
                         "refused": bool(res.refused), "book_ok": hit_book,
                         "page_ok": hit_page, "ms": round(ms, 1),
                         "lane": bool(res.channel_counts.get("evidence_lane"))})
    finally:
        loader.close()

    def pct(sel, key):
        sub = [r for r in rows if sel(r)]
        return round(100 * sum(1 for r in sub if r[key]) / len(sub), 1) if sub else None

    kinds = sorted({r["kind"] for r in rows})
    report = {
        "golden_set": str(args.golden), "domain": args.domain, "top_k": args.top,
        "pairs": len(rows),
        "page_recall_pct": pct(lambda r: True, "page_ok"),
        "book_recall_pct": pct(lambda r: True, "book_ok"),
        "refused_pct": round(100 * sum(1 for r in rows if r["refused"]) / len(rows), 1),
        "by_kind": {k: {"n": sum(1 for r in rows if r["kind"] == k),
                        "page_recall_pct": pct(lambda r, k=k: r["kind"] == k, "page_ok"),
                        "book_recall_pct": pct(lambda r, k=k: r["kind"] == k, "book_ok")}
                    for k in kinds},
        "by_lang": {l: pct(lambda r, l=l: r["lang"] == l, "page_ok")
                    for l in sorted({r["lang"] for r in rows if r["lang"]})},
        "latency_ms_mean": round(statistics.mean(r["ms"] for r in rows), 1),
        "latency_ms_p95": round(sorted(r["ms"] for r in rows)[int(len(rows) * 0.95)], 1),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"report": report, "rows": rows},
                                    indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
