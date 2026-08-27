#!/usr/bin/env python3
"""How much of the corpus the ingredient registry can actually read.

R1's exit criterion is a measured number, not the length of a list: the
registry is finished when it resolves >=95% of the corpus's quantity lines, and
the lines it cannot read are REPORTED rather than ignored — an unknown line is
a gap in the vocabulary, and a vocabulary that hides its gaps stops growing.

A "quantity line" is any line of a page that states a number against a unit the
registry knows. That is deliberately mechanical: it does not presuppose the
line is an ingredient, so prose like "bake for 20 minutes" is counted against
us and the figure stays honest.

    python tools/mine_ingredients.py --domain canon_library
    python tools/mine_ingredients.py --domain canon_library --top 60   # the gaps
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from graphify_ent.loader import Neo4jLoader
from graphify_ent.recipes.ingredients import Registry, norm, resolve_line

#: Recipe metadata that states a quantity without being an ingredient row:
#: "Total weight: 2 lb 14 oz", "Yield: two pies", "Scale at 20 oz". No
#: ingredient registry should be blamed for these — but they are COUNTED and
#: reported, never silently dropped, exactly like bench_canon's unlocatable.
_METADATA = __import__("re").compile(
    r"total weight|approximate weight|yield|makes\b|scale at|test\b|"
    r"desired dough temperature|pour \d+ personnes|per \d+ (persone|porzioni)",
    __import__("re").I)

#: Words that head a line without naming a thing — reporting them as vocabulary
#: gaps would send the registry chasing prose.
_STOP = set("""the a an of and or with for from to in on at by is are was were
this that these those each about approximately per about into until then
add mix stir cook bake heat place pour remove cover serve using use make
il lo la i gli le un uno una di del della dei delle e con per da in su
le la les un une des du de et avec pour dans sur par""".split())


def head_words(line: str, n: int = 3) -> str:
    """The line minus its figures — what a human would call the row."""
    body = re.sub(r"\d+[^a-zà-ÿ]*", " ", norm(line))
    words = [w for w in re.findall(r"[a-zà-ÿ']{3,}", body) if w not in _STOP]
    return " ".join(words[:n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="canon_library")
    ap.add_argument("--records", type=Path, default=None,
                    help="limita la misura alle pagine di riferimento di questi "
                         "abbinamenti (il metro di R5): il registro serve "
                         "all'abbinamento, non a leggere ogni riga di ogni libro")
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--top", type=int, default=40, help="how many gaps to list")
    ap.add_argument("--json", type=Path,
                    default=Path("../evidence/T100/ingredient-coverage.json"))
    args = ap.parse_args()

    reg = Registry.load(args.registry)
    units = sorted(set(reg.mass_g) | set(reg.volume_ml), key=len, reverse=True)
    qty_line = re.compile(rf"\d+\s*(?:{'|'.join(re.escape(u) for u in units)})(?![a-z])")

    scope = None
    if args.records:
        recs = json.loads(args.records.read_text())["records"]
        scope = {(r["corpus_file"], int(r["gt_pdf_page"]))
                 for r in recs if r.get("in_corpus") and r.get("gt_pdf_page")}

    loader = Neo4jLoader()
    seen = resolved = metadata = 0
    per_book: dict[str, list[int]] = {}
    classes: Counter = Counter()
    gaps: Counter = Counter()
    from graphify_ent.recipes.ingredients import parse_block
    with loader._session() as s:
        rows = s.run(
            "MATCH (n:Entity) WHERE n.domain = $d AND n.extraction_method = 'page' "
            "RETURN n.source_file AS f, n.page_lo AS pg, n.passage AS p",
            d=args.domain)
        for row in rows:
            if scope is not None and (row["f"], row["pg"]) not in scope:
                continue
            book = (row["f"] or "?")[:34]
            book_stats = per_book.setdefault(book, [0, 0])
            # Rows, not raw lines. A two-column book prints "180 g" on a line
            # of its own: counting that line as an unresolved ingredient is an
            # artefact of the measure, not a gap in the registry — measured,
            # it put Gisslen at 7.3% while his rows parse cleanly. parse_block
            # regroups the columns; a quantity line whose row resolved is
            # covered, one whose row did not is a genuine gap.
            page = row["p"] or ""
            covered_rows = parse_block(page, reg)
            covered_raw = " ".join(r.raw for r in covered_rows)
            for r in covered_rows:
                classes[r.cls] += 1
            lines = [x.strip() for x in page.splitlines() if x.strip()]
            for i, line in enumerate(lines):
                if len(line) < 3 or not qty_line.search(norm(line)):
                    continue
                # A metadata row spans columns like an ingredient row does, so
                # the label may sit up to TWO lines above ("Total weight:" /
                # "2 lb 14 oz" / "1258 g" — the metric column is the third
                # line of the row).
                ctx = " ".join(lines[max(0, i - 2): i + 1])
                if _METADATA.search(norm(ctx)):
                    metadata += 1
                    continue
                seen += 1
                book_stats[0] += 1
                if norm(line) in norm(covered_raw):
                    resolved += 1
                    book_stats[1] += 1
                else:
                    head = head_words(line)
                    if head:
                        gaps[head] += 1
    loader.close()

    pct = round(100 * resolved / max(seen, 1), 1)
    report = {
        "domain": args.domain,
        "scope": ("reference pages of " + str(args.records)
                  if args.records else "whole corpus"),
        "scope_pages": len(scope) if scope else None,
        "registry_version": reg.version,
        "canonical_ingredients": len(reg.ingredients),
        "quantity_lines": seen,
        "metadata_lines_excluded": metadata,
        "resolved": resolved,
        "coverage_pct": pct,
        "target_pct": 95.0,
        "meets_target": pct >= 95.0,
        "by_class": dict(classes),
        "by_book": {b: {"lines": v[0], "resolved": v[1],
                        "coverage_pct": round(100 * v[1] / max(v[0], 1), 1)}
                    for b, v in sorted(per_book.items())},
        "top_gaps": [{"head": h, "lines": c} for h, c in gaps.most_common(args.top)],
        "gap_lines_total": sum(gaps.values()),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "top_gaps"},
                     indent=2, ensure_ascii=False))
    print("\nvocaboli mancanti piu' frequenti:")
    for g in report["top_gaps"]:
        print(f"  {g['lines']:6}  {g['head']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
