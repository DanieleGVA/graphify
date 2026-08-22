#!/usr/bin/env python3
"""Phase 0.3 — quantify silent character loss in the stock extraction path.

For every PDF under a corpus root, compare the full extracted text length
against what `graphify.llm._read_files` would actually dispatch to the model
(`_FILE_CHAR_CAP = 20_000` per file, hard truncation, no slicing in v8).

The reported percentage is the before/after KPI for Phase 1.1: after PdfSlice
lands, `--after` re-runs the same comparison against the slicing path and must
report < 1 % dropped.

Usage:
    python tools/audit_truncation.py <corpus-root> [--after] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def full_text_len(path: Path) -> tuple[int, int]:
    """Return (chars, pages) of the complete PDF text via PyMuPDF, pypdf fallback."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(str(path)) as doc:
            return sum(len(p.get_text()) for p in doc), doc.page_count
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return sum(len(p.extract_text() or "") for p in reader.pages), len(reader.pages)


def dispatched_len_stock(path: Path, cap: int) -> int:
    """Chars the stock v8 path would dispatch: one truncated read per file."""
    total, _ = full_text_len(path)
    return min(total, cap)


def dispatched_len_sliced(path: Path, cap: int) -> int:
    """Chars the Phase 1.1 slicing path dispatches: sum over page-range slices."""
    from graphify_ent.file_slice import read_slice_text, slice_pdf

    return sum(len(read_slice_text(s)) for s in slice_pdf(path, cap))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--after", action="store_true", help="measure the PdfSlice path")
    ap.add_argument("--cap", type=int, default=None, help="override char cap")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.cap is None:
        from graphify.llm import _FILE_CHAR_CAP

        args.cap = _FILE_CHAR_CAP

    pdfs = sorted(p for p in args.root.rglob("*.pdf") if p.is_file())
    if not pdfs:
        print(f"no PDFs under {args.root}", file=sys.stderr)
        return 1

    rows, total_full, total_sent = [], 0, 0
    for p in pdfs:
        full, pages = full_text_len(p)
        sent = (dispatched_len_sliced if args.after else dispatched_len_stock)(p, args.cap)
        dropped = max(full - sent, 0)
        rows.append(
            {
                "file": p.name,
                "pages": pages,
                "full_chars": full,
                "dispatched_chars": sent,
                "dropped_chars": dropped,
                "dropped_pct": round(100 * dropped / full, 2) if full else 0.0,
            }
        )
        total_full += full
        total_sent += sent
        print(
            f"{rows[-1]['dropped_pct']:>6.2f}%  {p.name[:58]:<58} "
            f"{pages:>5}pp  full={full:>9,}  sent={sent:>9,}"
        )

    total_dropped = max(total_full - total_sent, 0)
    pct = round(100 * total_dropped / total_full, 2) if total_full else 0.0
    mode = "PdfSlice (after)" if args.after else "stock v8 (before)"
    print(f"\n{'=' * 78}\nMode: {mode}   cap={args.cap:,} chars/file")
    print(f"PDFs: {len(pdfs)}   total full chars: {total_full:,}   dispatched: {total_sent:,}")
    print(f"TOTAL {pct}% of characters silently dropped")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"mode": mode, "cap": args.cap, "files": rows,
                 "total_full_chars": total_full, "total_dispatched_chars": total_sent,
                 "total_dropped_chars": total_dropped, "dropped_pct": pct},
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
