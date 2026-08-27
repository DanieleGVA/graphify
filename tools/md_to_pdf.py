#!/usr/bin/env python3
"""Turn an OCR transcription back into a paginated, text-bearing PDF.

`ocr_book.py` writes Markdown with a `<!-- page N -->` marker per page. The
ingest pipeline reads PDFs and epubs through PyMuPDF, so rather than teach
every stage a third format, the transcription is rebuilt as a PDF whose page N
holds the text of page N of the original.

That equivalence is the point. A citation the graph returns as "page 217" then
means page 217 of the book on the shelf, exactly as for a natively extractable
book — the OCR step becomes invisible to everything downstream, including the
verifier, and provenance survives the round trip.

    python tools/md_to_pdf.py ../evidence/T91/ducasse.md --out ../canon_library/Ducasse.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MARKER = re.compile(r"<!--\s*page\s+(\d+)\s*-->")

#: The built-in PDF fonts cover Latin-1, so a typographic apostrophe or an
#: en dash comes out as "?" — measured: "l'Ecotais" became "l?Ecotais". The
#: characters are folded to their Latin-1 equivalents rather than dropped: the
#: retrieval layer folds punctuation anyway, and a "?" inside a word does not.
_TYPOGRAPHIC = {
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
    "\u0153": "oe", "\u0152": "OE", "\u2032": "'", "\u02bc": "'",
}


def latin1_safe(text: str) -> str:
    for a, b in _TYPOGRAPHIC.items():
        text = text.replace(a, b)
    # Anything still outside Latin-1 would render as "?"; drop it rather than
    # let it corrupt a word.
    return text.encode("latin-1", "ignore").decode("latin-1")


#: A blank page in the transcription must stay blank here: a page of plates
#: carries no text, and inventing a placeholder would make it look extractable.
PLACEHOLDER = re.compile(r"^\*\[pagina \d+: nessun testo\]\*\s*$", re.M)


def pages_from_markdown(text: str) -> dict[int, str]:
    """Page number → its text. Anything before the first marker is dropped:
    it is the file's own header, not a page of the book."""
    out: dict[int, str] = {}
    parts = MARKER.split(text)
    for i in range(1, len(parts) - 1, 2):
        n = int(parts[i])
        body = latin1_safe(PLACEHOLDER.sub("", parts[i + 1]).strip())
        out[n] = body
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--fontsize", type=float, default=9.0)
    args = ap.parse_args()

    import fitz

    pages = pages_from_markdown(args.markdown.read_text(encoding="utf-8"))
    if not pages:
        print("nessun marcatore di pagina trovato: il markdown non viene da ocr_book.py")
        return 1

    doc = fitz.open()
    base_w, base_h = fitz.paper_size("a4")
    margin = 40
    enlarged = 0
    for n in range(1, max(pages) + 1):
        body = pages.get(n, "")
        if not body:
            doc.new_page(width=base_w, height=base_h)
            continue
        # One markdown page becomes exactly one PDF page — that equivalence is
        # what keeps "page 217" meaning page 217 of the book. So when a dense
        # page will not fit, GROW the sheet rather than shrink the text past
        # legibility or split the page in two: measured, 11 of 438 Ducasse
        # pages overflowed A4 even at 4 pt, and splitting them would have
        # renumbered everything after.
        for scale in (1.0, 1.4, 2.0, 3.0):
            w, h = base_w * scale, base_h * scale
            page = doc.new_page(width=w, height=h)
            rect = fitz.Rect(margin, margin, w - margin, h - margin)
            size = args.fontsize
            placed = False
            while size >= 5.0:
                if page.insert_textbox(rect, body, fontsize=size, fontname="helv") >= 0:
                    placed = True
                    break
                size -= 1.0
            if placed:
                if scale > 1.0:
                    enlarged += 1
                break
            doc.delete_page(doc.page_count - 1)
        else:
            page = doc.new_page(width=base_w * 3, height=base_h * 3)
            page.insert_textbox(fitz.Rect(margin, margin, base_w * 3 - margin,
                                          base_h * 3 - margin),
                                body, fontsize=5.0, fontname="helv")
    if args.title:
        doc.set_metadata({"title": args.title})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(args.out), garbage=4, deflate=True)
    doc.close()

    with fitz.open(args.out) as check:
        total_pages = check.page_count
        got = sum(1 for p in check if p.get_text().strip())
        chars = sum(len(p.get_text()) for p in check)
    expected = sum(1 for v in pages.values() if v)
    print(f"pagine scritte {total_pages} · con testo {got}/{expected} attese · "
          f"{chars:,} caratteri · fogli ingranditi {enlarged}")
    if got < expected:
        print(f"ATTENZIONE: {expected - got} pagine del markdown non hanno prodotto testo")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
