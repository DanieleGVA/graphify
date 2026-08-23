#!/usr/bin/env python3
"""Build a question set that belongs to neither extraction.

`golden-qa-v1.json` takes its queries from the structural graph's own node
text. Whatever metric you compute on it, the structural graph gets an exact
fulltext match for free and every other extraction is answering a question
phrased in a vocabulary it never produced. Anchoring the page metric to that
set does not fix it: the bias is in the QUERIES, not in the ground-truth key.

So draw the queries from the source PDFs directly, via PyMuPDF, page by page.
Neither graph contributed a word. Ground truth is the page the terms came
from, which every extraction records independently.

Terms are chosen to be distinctive: alphabetic, long enough to be meaningful,
and rare across the corpus, so a query identifies a passage rather than
matching everywhere. Pages are sampled deterministically (fixed stride, no RNG)
so the set is reproducible.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import fitz

_WORD = re.compile(r"[A-Za-zÀ-ÿ]{5,}")
STOP = {
    "which", "there", "these", "those", "their", "about", "would", "could",
    "should", "other", "after", "before", "while", "where", "being", "under",
    "dans", "avec", "pour", "cette", "leurs", "comme", "plus", "sont", "elle",
    "della", "delle", "degli", "questo", "questa", "sono", "come", "anche",
}


#: 4+ consecutive consonants is the signature of OCR damage in these scans
#: ("cnoosticks", "rinegor"). Selecting the RAREST terms selects exactly that
#: noise, which would hand the advantage to whichever extraction preserves raw
#: scan artefacts — the opposite of neutral. Reject them.
_GARBLED = re.compile(r"[bcdfghjklmnpqrstvwxz]{4,}", re.I)


def page_terms(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)
            if w.lower() not in STOP and not _GARBLED.search(w)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("../pilot"))
    ap.add_argument("--per-book", type=int, default=25)
    ap.add_argument("--terms", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("../eval/neutral-qa-v1.json"))
    args = ap.parse_args()

    pdfs = sorted(args.corpus.glob("*.pdf"))
    pages: dict[str, list[tuple[int, str]]] = {}
    df = collections.Counter()          # in how many pages each term appears

    for pdf in pdfs:
        doc = fitz.open(pdf)
        got = []
        for i in range(doc.page_count):
            t = doc[i].get_text()
            if len(t) < 900:            # skip plates, blanks, index stubs
                continue
            got.append((i + 1, t))
            df.update(set(page_terms(t)))
        doc.close()
        pages[pdf.name] = got

    total_pages = sum(len(v) for v in pages.values())
    pairs = []
    for fname, got in pages.items():
        if not got:
            continue
        stride = max(1, len(got) // args.per_book)
        for pageno, text in got[::stride][: args.per_book]:
            terms = page_terms(text)
            if len(set(terms)) < 12:
                continue
            # rarest first: a term on 4000 pages identifies nothing
            # Mid-frequency band: rare enough to identify a passage, common
            # enough to be a real word rather than a one-off scan artefact.
            band = [w for w in set(terms) if 3 <= df[w] <= 40]
            uniq = sorted(band, key=lambda w: (-len(w), df[w]))
            picked = uniq[: args.terms]
            if len(picked) < args.terms:
                continue
            query = " ".join(picked)
            pairs.append({
                "id": "n_" + hashlib.sha1(f"{fname}{pageno}{query}".encode()).hexdigest()[:10],
                "query": query,
                "query_lang": "xx",
                "kind": "monolingual",
                "ground_truth_doc": fname,
                "ground_truth_node": None,
                "answer_span": "",
                "source_location": f"pages {pageno}-{pageno}",
            })

    out = {
        "version": "neutral-v1",
        "corpus": [p.name for p in pdfs],
        "provenance": "terms drawn from the source PDFs via PyMuPDF, page by page; "
                      "no extraction contributed to the queries. Ground truth is the "
                      "page of origin, which every extraction records independently.",
        "counts": {"pairs": len(pairs), "pages_considered": total_pages},
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{len(pairs)} domande da {total_pages} pagine -> {args.out}")
    for p in pairs[:5]:
        print(f"   {p['ground_truth_doc'][:26]:<28} p.{p['source_location'][6:]:<10} "
              f"{p['query']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
