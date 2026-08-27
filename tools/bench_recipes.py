#!/usr/bin/env python3
"""R5 — the number that counts: the recipe matcher against 234 human matches.

The canon report is a ready-made benchmark: a human validator matched every
board card to a published reference and quoted it. This measures whether the
fingerprint matcher finds the same page — recipe identity by ingredients,
procedure and title, per the T96 direction — with **T96's untuned weights**.
No calibration has happened, so there is nothing to hold out: the moment a
weight is tuned, the per-book split becomes mandatory (plan §R5) and this
header is the reminder.

Cards are joined to records by their QUOTE — each record's quote is printed on
its card page in the report — because titles differ ("Nachos" the record,
"Beef Nachos" the card) and joining on them mismatched.

Two jobs, mirroring bench_canon:
  * 200 in-corpus records: the human's page must rank first (top-1) or among
    the first three (top-3) of the whole corpus index. A hit is same book and
    within one page — recipes straddle page breaks.
  * 34 out-of-corpus records: the matcher must not produce a confident match —
    scores are reported so the separation between true matches and
    out-of-corpus best-hits is measured, not asserted.

    python tools/bench_recipes.py --domain canon_library
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import fitz

from graphify_ent.loader import Neo4jLoader
from graphify_ent.recipes.ingredients import Registry
from graphify_ent.recipes.match import CorpusIndex, RecipeQuery

LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "­": ""}


def squash(s: str) -> str:
    for k, v in LIG.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", "", s).lower()


def load_cards(report: Path) -> list[dict]:
    """The report's cards, delimited by their own structure — never by page.

    A card is not a page: cards run over PDF page breaks, and reading one page
    as one card handed the Paris-Brest a query full of another card's carrots
    and mushrooms. Every card opens with its dish name on the line before
    "Board name:", so the report is split there; the ingredient block runs
    from "Ingredients" to "Method:" (or the card's end), which also keeps the
    reference QUOTE — printed before the block — out of the query.
    """
    with fitz.open(report) as doc:
        full = "\n".join(doc[i].get_text() for i in range(doc.page_count))
    marks = [m.start() for m in re.finditer(r"^Board name:", full, re.M)]
    cards = []
    for i, at in enumerate(marks):
        head = full[:at].rstrip().splitlines()
        title = head[-1].strip() if head else ""
        end = marks[i + 1] if i + 1 < len(marks) else len(full)
        # the next card's title is the last line before its marker: cut it off
        chunk = full[at:end]
        if i + 1 < len(marks):
            chunk = chunk.rsplit("\n", 2)[0]
        if "Ingredients" not in chunk:
            continue
        body = chunk.split("Ingredients", 1)[1]
        # everything after the HACCP block is storage temperatures and
        # allergens — quantities that are not ingredients
        body = re.split(r"HACCP PARAMETERS", body)[0]
        if "Method:" in body:
            body = body.split("Method:", 1)[0] + "Method:" +                 body.split("Method:", 1)[1]
        cards.append({"title": title, "body": body, "squash": squash(chunk)})
    return cards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("../eval/canon/records-canon.json"))
    ap.add_argument("--report", type=Path,
                    default=Path("../tests/CANON_VALIDATED_RECIPES_REPORT_v001.pdf"))
    ap.add_argument("--domain", default="canon_library")
    ap.add_argument("--cache", type=Path, default=Path("../evidence/T100/corpus-index.json"))
    ap.add_argument("--json", type=Path, default=Path("../evidence/T100/bench-recipes.json"))
    ap.add_argument("--page-tolerance", type=int, default=1)
    args = ap.parse_args()

    records = json.loads(args.records.read_text())["records"]
    cards = load_cards(args.report)
    loader = Neo4jLoader()
    t0 = time.perf_counter()
    index = CorpusIndex.from_graph(loader, args.domain, cache=args.cache)
    loader.close()
    build_s = round(time.perf_counter() - t0, 1)

    reg = Registry.load()
    rows, misses = [], 0
    for rec in records:
        probe = squash(rec["quote"])[:80]
        card = next((c for c in cards if probe and probe in c["squash"]), None)
        if card is None:
            misses += 1
            continue
        t0 = time.perf_counter()
        q = RecipeQuery.from_text(card["body"], title=card["title"], registry=reg)
        ranked = index.rank(q)[:10]
        ms = (time.perf_counter() - t0) * 1000
        want_book = rec.get("corpus_file") or ""
        want_page = rec.get("gt_pdf_page")
        rank = None
        if rec.get("in_corpus") and want_page:
            for pos, m in enumerate(ranked, 1):
                if (m.candidate.source_file == want_book
                        and abs(m.candidate.page - int(want_page)) <= args.page_tolerance):
                    rank = pos
                    break
        rows.append({
            "dish": rec["dish"][:60], "book": rec["book_key"],
            "match": rec.get("match"), "in_corpus": rec["in_corpus"],
            "gt_page": want_page, "rank": rank,
            "top1": rank == 1, "top3": rank is not None and rank <= 3,
            "best_score": round(ranked[0].combined, 4) if ranked else 0.0,
            "best": ranked[0].as_dict() if ranked else None,
            "n_ingredients": len(q.proportions), "ms": round(ms, 1),
        })

    inc = [r for r in rows if r["in_corpus"] and r["gt_page"]]
    out = [r for r in rows if not r["in_corpus"]]
    strong = [r for r in inc if r["match"] == "strong"]

    def pct(rows_, key):
        return round(100 * sum(1 for r in rows_ if r[key]) / max(len(rows_), 1), 1)

    def mrr(rows_):
        return round(statistics.mean(
            (1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows_), 3) if rows_ else 0.0

    true_scores = sorted(r["best_score"] for r in inc if r["top1"])
    out_scores = sorted(r["best_score"] for r in out)
    report = {
        "records": len(records), "cards_matched": len(rows), "cards_missing": misses,
        "index": {"candidates": len(index.candidates), "build_s": build_s},
        "weights": "T96 untuned (0.60/0.25/0.15) — no calibration has happened",
        "in_corpus": {
            "n": len(inc),
            "top1_pct": pct(inc, "top1"), "top3_pct": pct(inc, "top3"),
            "mrr": mrr(inc),
            "strong": {"n": len(strong), "top1_pct": pct(strong, "top1"),
                       "top3_pct": pct(strong, "top3")},
            "by_book": {b: f"{sum(1 for r in inc if r['book'] == b and r['top1'])}"
                           f"/{sum(1 for r in inc if r['book'] == b)}"
                        for b in sorted({r["book"] for r in inc})},
        },
        "out_of_corpus": {
            "n": len(out),
            "best_scores": {"min": out_scores[0] if out_scores else None,
                            "median": round(statistics.median(out_scores), 4) if out_scores else None,
                            "max": out_scores[-1] if out_scores else None},
        },
        "true_match_scores": {"min": true_scores[0] if true_scores else None,
                              "median": round(statistics.median(true_scores), 4) if true_scores else None},
        "latency_ms_mean": round(statistics.mean(r["ms"] for r in rows), 1),
        "targets": {"top1_strong": 80.0, "top3": 90.0},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"report": report, "rows": rows},
                                    indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
