#!/usr/bin/env python3
"""R5 on the new test input: the Pareto cards against their declared canon.

The Pareto export replaced the validated-recipes report as the test file
(Daniele, 2026-08-27), and the ground truth it carries is a different shape.
The old report gave a human's page-level match — book AND page — for 234
records. This one gives, per card, the reference WORK:

    Canon  CIA Professional Chef          -> the corpus holds it
    Canon  Italian canon — TO ACQUIRE     -> the corpus does not

So the question this benchmark can answer honestly is **book-level**: does the
matcher return a page from the book the card says it follows? Page-level top-1
against a human is no longer measurable from this file — that is a real loss
and it is reported here rather than papered over with a proxy.

What the file adds in exchange is the negative case, stated by the document
itself: 4 of the 17 cards follow a work the corpus does not hold, and a matcher
that returns a confident page for those is manufacturing a reference. Their
scores are reported next to the true ones so the separation is measured.

    python tools/bench_cards_pareto.py --domain canon_library
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from graphify_ent.loader import Neo4jLoader
from graphify_ent.recipes.cards import load_cards
from graphify_ent.recipes.ingredients import Registry, proportions
from graphify_ent.recipes.match import CorpusIndex, RecipeQuery
from graphify_ent.recipes.techniques import techniques_in

#: Which corpus file each `Canon` value refers to. Declared here rather than
#: guessed from the string: "CIA Professional Chef" is a house name for a book
#: whose file is called something else, and a fuzzy match would quietly count
#: the wrong book as a hit.
CANON_BOOKS = {
    "CIA Professional Chef": [
        "The Professional Chef - The Culinary Institute of America.pdf"],
    "Pastry: Gisslen / Ducasse": [
        "Professional Baking (Wayne Gisslen).pdf",
        "Le Grand Livre de cuisine - Desserts et Patisseries (Alain Ducasse).pdf"],
}


def books_for(canon: str, available: list[str]) -> list[str]:
    want = CANON_BOOKS.get((canon or "").strip(), [])
    return [b for b in want if b in available]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path,
                    default=Path("../tests/Pareto_Recipe_Cards_v001_abstract.pdf"))
    ap.add_argument("--domain", default="canon_library")
    ap.add_argument("--cache", type=Path,
                    default=Path("../evidence/T101/corpus-index.json"))
    ap.add_argument("--json", type=Path,
                    default=Path("../evidence/T101/bench-cards-pareto.json"))
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    reg = Registry.load()
    cards = load_cards(args.cards)
    loader = Neo4jLoader()
    t0 = time.perf_counter()
    index = CorpusIndex.from_graph(loader, args.domain, registry=reg, cache=args.cache)
    loader.close()
    build_s = round(time.perf_counter() - t0, 1)
    available = sorted({c.source_file for c in index.candidates})

    rows = []
    for card in cards:
        resolved = card.resolved(reg)
        props = proportions(resolved)
        method = card.procedure
        query = RecipeQuery(title=card.title, resolved=resolved, proportions=props,
                            verbs=techniques_in(method),
                            verb_seq=techniques_in(method, ordered=True))
        t0 = time.perf_counter()
        ranked = index.rank(query)[: args.top]
        ms = (time.perf_counter() - t0) * 1000
        want = books_for(card.canon, available)
        rank = next((i for i, m in enumerate(ranked, 1)
                     if m.candidate.source_file in want), None) if want else None
        rows.append({
            "number": card.number, "title": card.title[:70], "canon": card.canon,
            "canon_available": card.canon_available,
            "canon_files_in_corpus": want,
            "lines": len(card.lines),
            "recognised": len(resolved),
            "quantified": sum(1 for r in resolved if r.quantified),
            "book_rank": rank, "book_top1": rank == 1,
            "book_top3": rank is not None and rank <= 3,
            "best_score": round(ranked[0].combined, 4) if ranked else 0.0,
            "best": ranked[0].as_dict() if ranked else None,
            "ms": round(ms, 1),
        })

    scored = [r for r in rows if r["canon_files_in_corpus"]]
    negatives = [r for r in rows if not r["canon_available"]]
    unmapped = [r for r in rows if r["canon_available"] and not r["canon_files_in_corpus"]]
    lines_total = sum(r["lines"] for r in rows)

    def pct(rows_, key):
        return round(100 * sum(1 for r in rows_ if r[key]) / max(len(rows_), 1), 1)

    report = {
        "cards": len(rows),
        "index": {"candidates": len(index.candidates), "build_s": build_s},
        "parse": {
            "ingredient_lines": lines_total,
            "recognised": sum(r["recognised"] for r in rows),
            "recognised_pct": round(100 * sum(r["recognised"] for r in rows)
                                    / max(lines_total, 1), 1),
            "quantified": sum(r["quantified"] for r in rows),
        },
        "ground_truth": "book-level only — this export carries no page reference",
        "book_level": {
            "n": len(scored),
            "top1_pct": pct(scored, "book_top1"),
            "top3_pct": pct(scored, "book_top3"),
            "per_card": {r["title"][:44]: r["book_rank"] for r in scored},
        },
        "negatives_to_acquire": {
            "n": len(negatives),
            "titles": [r["title"][:44] for r in negatives],
            "best_scores": sorted(r["best_score"] for r in negatives),
        },
        "canon_not_mapped": [r["canon"] for r in unmapped],
        "true_scores": sorted(r["best_score"] for r in scored if r["book_top1"]),
        "latency_ms_mean": round(statistics.mean(r["ms"] for r in rows), 1),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"report": report, "rows": rows},
                                    indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
