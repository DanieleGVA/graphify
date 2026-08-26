#!/usr/bin/env python3
"""T81 — the canon report as a test set: 234 recipes, two jobs, one graph.

The report (`tests/CANON_VALIDATED_RECIPES_REPORT_v001.pdf`) grounds every
recipe with a verbatim quote from a published reference and its page. That
makes it a ready-made benchmark with externally-authored ground truth:

  * **107 in-corpus records** (Professional Chef, Larousse — the loaded books):
    the system must hand back the quoted passage, book and page. Success is
    containment of the quote in a returned passage, squash-normalized, because
    both the report and the graph inherited PDF hyphenation artifacts
    ("heavy- gauge", "T omato") that no reader sees.
  * **127 out-of-corpus records** (Professional Baking, American Regional, …):
    the system must NOT produce grounding — a quote from a book the corpus
    does not hold, confirmed anywhere, would be manufactured evidence.

Queries go through the same `HybridRetriever` the component serves; nothing is
tuned per record. Ground truth (`gt_pdf_page`) was established against the
source PDFs by the extraction step in the scaffolding repo, never from the
graph.
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

LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
       "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st", "­": ""}


def squash(s: str) -> str:
    """Case-, space- and hyphenation-artifact-insensitive form."""
    for k, v in LIG.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", "", s).lower()


def content_words(quote: str, n: int = 4) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ]{5,}", quote)[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("../eval/canon/records.json"))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--domain", default="pilot")
    ap.add_argument("--json", type=Path, default=Path("../evidence/T81/bench-canon.json"))
    args = ap.parse_args()

    records = json.loads(args.records.read_text())["records"]
    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    embedder.encode(["warm up"])
    enc = lambda q: embedder.encode([q])[0]  # noqa: E731

    rows = []
    try:
        for e in records:
            query = f"{e['dish']} {' '.join(content_words(e['quote']))}"
            t0 = time.perf_counter()
            res = retriever.query(query, embed_fn=enc,
                                  channels=("vector", "fulltext", "graph"),
                                  hops=1, domain=args.domain)
            ids = [h.node_id for h in res.hits[: args.top]]
            props = retriever.hydrate(ids) if ids else {}
            q = squash(e["quote"])
            grounded, where, page_ok = False, None, False
            for nid in ids:
                p = props.get(nid) or {}
                body = squash((p.get("passage") or "") + " " +
                              (p.get("text_excerpt") or ""))
                if q in body or (len(q) > 120 and q[:120] in body):
                    grounded = True
                    where = f"{(p.get('source_file') or '')[:24]} {p.get('source_location') or ''}"
                    m = re.search(r"(\d+)\s*-\s*(\d+)", p.get("source_location") or "")
                    if m and e.get("gt_pdf_page"):
                        page_ok = int(m.group(1)) <= e["gt_pdf_page"] <= int(m.group(2))
                    break
            ms = (time.perf_counter() - t0) * 1000
            correct = grounded if e["in_corpus"] else not grounded
            rows.append({"dish": e["dish"][:60], "book": e["book_key"],
                         "page": e["page"], "verdict": e["verdict"],
                         "in_corpus": e["in_corpus"], "grounded": grounded,
                         "page_ok": page_ok, "correct": correct,
                         "refused": bool(res.refused), "ms": round(ms, 1),
                         "where": where})
    finally:
        loader.close()

    inc = [r for r in rows if r["in_corpus"]]
    out = [r for r in rows if not r["in_corpus"]]
    lat = [r["ms"] for r in rows]
    report = {
        "records": len(rows),
        "in_corpus": {
            "n": len(inc),
            "grounded": sum(r["grounded"] for r in inc),
            "grounding_pct": round(100 * sum(r["grounded"] for r in inc)
                                   / max(len(inc), 1), 1),
            "page_ok": sum(r["page_ok"] for r in inc),
            "by_book": {b: f"{sum(r['grounded'] for r in inc if r['book'] == b)}"
                           f"/{sum(1 for r in inc if r['book'] == b)}"
                        for b in sorted({r['book'] for r in inc})},
        },
        "out_of_corpus": {
            "n": len(out),
            "false_groundings": sum(r["grounded"] for r in out),
            "rejection_pct": round(100 * sum(not r["grounded"] for r in out)
                                   / max(len(out), 1), 1),
        },
        "latency_ms_mean": round(statistics.mean(lat), 1),
        "latency_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 1),
        "accuracy_pct": round(100 * sum(r["correct"] for r in rows) / len(rows), 1),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"report": report, "rows": rows},
                                    indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
