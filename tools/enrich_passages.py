#!/usr/bin/env python3
"""Give every node the paragraph its quote came from, and link it to its page.

Measured on the loaded graph: concept nodes carry **32 characters** of evidence
on average, passage nodes 397 (capped at 400), and there are **zero** edges
between the two. That combination is why a documentary check — "does this
recipe match the source?" — cannot be done with the system: asked for the roux
proportions, it returns `'clarified butter'` and `'sifted flour'` and drops the
300 g and 350 g, which are the only part that settles the question. It knows
WHERE, never WHAT.

Two deterministic repairs, no model calls, because every character needed is
already on disk:

  1. `passage` — the quote widened to the paragraph that contains it, so the
     numbers, the units and the sentence around them travel with the node.
     `evidence` is left untouched: the Q1 gate checks it is a literal substring
     of the node's source, and `passage` is now that source.

  2. `[:APPEARS_ON]` — concept → page edges, matched on book plus overlapping
     page range. This is what lets retrieval do "I found the concept, now give
     me the text around it", which is the step the recipe check needed and the
     graph could not take.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import fitz

from graphify_ent.loader import OVERLAP_MARK, Neo4jLoader

_PAGES = re.compile(r"(\d+)\s*-\s*(\d+)")
DEFAULT_WIDTH = 1200


_cache: dict[str, list[str]] = {}


def page_texts(pdf: Path) -> list[str]:
    key = str(pdf)
    if key not in _cache:
        if len(_cache) > 2:                     # books are large; keep few
            _cache.pop(next(iter(_cache)))
        doc = fitz.open(pdf)
        _cache[key] = [doc[i].get_text() for i in range(doc.page_count)]
        doc.close()
    return _cache[key]


def pages_of(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    m = _PAGES.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m = re.search(r"(\d+)", text)
    return (int(m.group(1)), int(m.group(1))) if m else None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def load_true_pages(paths: list[Path]) -> dict[str, tuple[int, int]]:
    """Map node id -> the page range its slice ACTUALLY covered.

    `parse_compact` takes `source_location` from a field the *model* fills in,
    and the model guesses: measured, 43% of concept nodes claim a page whose
    text does not contain their own quote. The extraction checkpoints record
    the real range per slice, so the truth is already on disk — and page
    provenance is not something to accept on a model's word when it can be
    measured.
    """
    out: dict[str, tuple[int, int]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pages = rec.get("pages")
            if not pages:
                continue
            span = (int(pages[0]), int(pages[1]))
            for n in rec.get("nodes") or []:
                nid = n.get("id")
                if nid:
                    out[nid] = span
    return out


def locate(evidence: str, texts: list[str], span: tuple[int, int]) -> tuple[int, int] | None:
    """Find the single page inside `span` whose text contains the quote."""
    ev = norm(evidence)
    if not ev:
        return None
    pattern = r"\s+".join(re.escape(w) for w in ev.split())
    lo, hi = max(1, span[0]), min(len(texts), span[1])
    for pageno in range(lo, hi + 1):
        if re.search(pattern, texts[pageno - 1], re.I):
            return (pageno, pageno)
    return None


def widen(evidence: str, source: str, width: int) -> str:
    """Return the paragraph around `evidence` inside `source`.

    Falls back to the head of the source when the quote cannot be located —
    a node keeping *some* real context beats a node keeping none, and the
    caller records which happened.
    """
    if not source:
        return ""
    ev = norm(evidence)
    if not ev:
        return source[:width]
    # Locate the quote in the ORIGINAL text. Extracted PDF text is full of
    # newlines mid-sentence, so match whitespace-insensitively rather than
    # literally — and take the real offset rather than estimating it from the
    # normalised one: measured, proportional mapping missed the window for 40%
    # of nodes and silently degraded them to "first 1200 characters of the page".
    pattern = r"\s+".join(re.escape(w) for w in ev.split())
    m = re.search(pattern, source, re.I)
    if not m:
        return source[:width]
    centre = (m.start() + m.end()) // 2
    lo = max(0, centre - width // 2)
    hi = min(len(source), lo + width)
    chunk = source[lo:hi]
    # trim to sentence-ish boundaries so the passage reads as prose
    first = chunk.find("\n")
    if 0 < first < 120:
        chunk = chunk[first + 1:]
    return chunk.strip()


def overlap_passage(page_text: str, next_text: str, width: int, next_page: int) -> str:
    """`page_text` plus the head of the following page, marked as borrowed.

    Built from the source text every run rather than appended to what is
    stored, so running the pass twice writes the same value — the alternative
    grows a passage without bound and nobody notices until a context blows up.
    """
    base = (page_text or "").strip()
    tail = re.sub(r"\s+", " ", next_text or "").strip()[: max(0, width)]
    if not tail or width <= 0:
        return base
    return base + OVERLAP_MARK.format(page=next_page) + tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("../pilot"))
    ap.add_argument("--domain", default=None,
                    help="arricchisce SOLO questo dominio. Senza filtro, in un grafo "
                         "multi-corpus i concetti di uno finiscono collegati alle "
                         "pagine dell'altro: stesso libro, dominio diverso.")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--page-width", type=int, default=6000,
                    help="page nodes carry their whole page, not a window")
    ap.add_argument("--link", action="store_true", help="also build APPEARS_ON edges")
    ap.add_argument("--overlap", type=int, default=0,
                    help="characters of the NEXT page appended to each page node "
                         "(0 = off). A recipe does not stop at the page break: "
                         "measured, 4 of 32 grounding failures were quotes that "
                         "straddle two pages, and no single page node contains "
                         "them. The borrowed text is marked and page_lo/page_hi "
                         "are left alone, so provenance still names one page.")
    ap.add_argument("--passage-methods", nargs="*", default=["page", "native"],
                    help="extraction_method values that identify the passage layer")
    ap.add_argument("--checkpoints", type=Path, nargs="*",
                    default=[Path("../evidence/T71/slices.jsonl"),
                             Path("../evidence/T71/slices-redo.jsonl")])
    ap.add_argument("--json", type=Path, default=Path("../evidence/T72/enrich-stats.json"))
    args = ap.parse_args()

    # Two failures lived in this one line. `*.pdf` skipped the four epub books
    # entirely — every node of theirs counted as "no_book" and kept its short
    # quote. And the keys were compared raw: macOS stores "Joël" decomposed
    # while the graph holds it composed, so 949 Larousse page nodes never found
    # their file. Both were silent: the run reported success either way.
    books = {}
    for pattern in ("*.pdf", "*.epub"):
        for p in args.corpus.glob(pattern):
            books[unicodedata.normalize("NFC", p.name)] = p
    truth = load_true_pages(args.checkpoints)
    print(f"pagine vere note per {len(truth):,} nodi (dai checkpoint di estrazione)")
    loader = Neo4jLoader()
    stats = {"submitted": 0, "applied": 0, "unapplied": 0, "widened": 0, "fallback": 0, "no_pages": 0, "no_book": 0,
             "scanned": 0, "edges": 0, "page_corrected": 0, "page_model_kept": 0,
             "overlapped": 0}
    t0 = time.perf_counter()

    with loader._session() as s:
        total = s.run("MATCH (n:Entity) WHERE $domain IS NULL OR n.domain = $domain "
                      "RETURN count(*) AS n", domain=args.domain).single()["n"]
        print(f"nodi da arricchire: {total:,}", flush=True)
        skip = 0
        while True:
            rows = list(s.run(
                "MATCH (n:Entity) WHERE $domain IS NULL OR n.domain = $domain "
                "RETURN n.id AS id, n.domain AS domain, n.source_file AS f, "
                "n.source_location AS loc, n.evidence AS ev, "
                "n.extraction_method AS m "
                "ORDER BY n.id SKIP $skip LIMIT $lim",
                skip=skip, lim=args.batch, domain=args.domain))
            if not rows:
                break
            updates = []
            for r in rows:
                stats["scanned"] += 1
                pdf = books.get(unicodedata.normalize("NFC", r["f"] or ""))
                if not pdf:
                    stats["no_book"] += 1
                    continue
                texts = page_texts(pdf)
                claimed = pages_of(r["loc"])
                span = truth.get(r["id"]) or claimed
                if not span:
                    stats["no_pages"] += 1
                    continue
                # Prefer the page that demonstrably contains the quote; fall
                # back to the slice's real range, and only then to what the
                # model claimed.
                exact = locate(r["ev"] or "", texts, span)
                pg = exact or span
                if exact and claimed and exact != claimed:
                    stats["page_corrected"] += 1
                elif not exact:
                    stats["page_model_kept"] += 1
                lo, hi = max(1, pg[0]), min(len(texts), pg[1])
                source = "\n".join(texts[lo - 1:hi])
                if (r["m"] or "") in args.passage_methods and r["m"] == "page":
                    # Built deterministically by build_pages.py, already
                    # carrying its whole page. Nothing to widen — but the page
                    # break is arbitrary and a quote can straddle it, so
                    # optionally carry the head of the next page as well.
                    if not args.overlap:
                        continue
                    nxt = texts[hi] if hi < len(texts) else ""
                    passage = overlap_passage(source, nxt, args.overlap, hi + 1)
                    if OVERLAP_MARK.format(page=hi + 1) not in passage:
                        continue                      # last page of the book
                    stats["overlapped"] += 1
                    updates.append({"id": r["id"], "domain": r["domain"],
                                    "p": passage, "lo": pg[0], "hi": pg[1]})
                    continue
                # A page node's job IS the page: giving it a 1,200-character
                # window around its own quote left ~3,300 characters of every
                # page in no node at all, so a fact stated elsewhere on the page
                # was unreachable. Concept nodes stay windowed — they are meant
                # to be one claim with its sentence, not a whole page.
                if (r["m"] or "") == "native":
                    passage = source[: args.page_width].strip()
                else:
                    passage = widen(r["ev"] or "", source, args.width)
                if not passage:
                    continue
                if norm(r["ev"] or "") and norm(r["ev"]) in norm(passage):
                    stats["widened"] += 1
                else:
                    stats["fallback"] += 1
                # `domain` travels with the row: the write matches on
                # (id, domain), and a row without it matches nothing — silently,
                # which is how a run reported 50,937 widened passages while the
                # database received 0.
                updates.append({"id": r["id"], "domain": r["domain"],
                                "p": passage, "lo": pg[0], "hi": pg[1]})
            if updates:
                applied = s.run(
                    "UNWIND $rows AS row "
                    "MATCH (n:Entity {id: row.id, domain: row.domain}) "
                    "SET n.passage = row.p, n.page_lo = row.lo, "
                    "n.page_hi = row.hi, "
                    "n.source_location = 'pages ' + toString(row.lo) + '-' "
                    "+ toString(row.hi) "
                    "RETURN count(n) AS n", rows=updates).single()["n"]
                # Submitted != applied is the silent data-loss path: a run once
                # reported 50,937 widened passages while the database received
                # none, because the rows lacked the key the MATCH needed.
                stats["submitted"] += len(updates)
                stats["applied"] += applied
                if applied != len(updates):
                    stats["unapplied"] += len(updates) - applied
            skip += args.batch
            print(f"  {min(skip, total):,}/{total:,}", flush=True)

        if args.link:
            print("\ncostruisco gli archi scheda -> pagina...", flush=True)
            # page_lo/page_hi were just written; index them or the join is a
            # 71,493 x 3,187 cross product.
            s.run("CREATE INDEX entity_pages IF NOT EXISTS "
                  "FOR (n:Entity) ON (n.source_file, n.page_lo)")
            s.run("CALL db.awaitIndexes(300)")
            # Which method IS the passage layer is a property of the data, not
            # a constant to hardcode: it was 'native' for the structural
            # extraction and is 'page' for build_pages.py. Ask the graph.
            res = s.run("""
                MATCH (c:Entity) WHERE NOT c.extraction_method IN $passage
                  AND ($domain IS NULL OR c.domain = $domain)
                MATCH (p:Entity) WHERE p.extraction_method IN $passage
                  AND p.source_file = c.source_file
                  AND p.domain = c.domain
                  AND p.page_lo <= c.page_hi AND c.page_lo <= p.page_hi
                MERGE (c)-[e:APPEARS_ON]->(p)
                RETURN count(e) AS n
            """, passage=args.passage_methods, domain=args.domain).single()
            stats["edges"] = res["n"] if res else 0

    stats["seconds"] = round(time.perf_counter() - t0, 1)
    stats["reconciled"] = stats["unapplied"] == 0
    loader.close()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
