"""Phase 1.1 — page-based slicing for PDFs (the Tier-1 truncation fix).

Upstream v8 has no slicing at all: `graphify.llm._read_files` truncates every
file at `_FILE_CHAR_CAP` (20,000 chars) and dispatches the head. On the pilot
corpus that silently drops 99.1 % of all characters (evidence/T00).

This module introduces a **unit** abstraction that flows through the existing
packer unchanged:

    unit := Path            # whole file, the upstream behaviour
          | PdfSlice        # a contiguous page range of one PDF

A PDF whose full text exceeds the cap is split into contiguous page ranges,
each ≤ cap where page granularity allows. Every slice reports the *parent* PDF
as `source_file`, so nodes still merge by source document exactly as before.

Design note (ADR-0003): the execution plan assumed a `graphify/file_slice.py`
with a char-range `FileSlice` to mirror. That module does not exist in v8, so
the dataclass/parent-path/boundary design is established here instead, and the
enterprise code lives under `graphify_ent/` per the CLAUDE.md fork-scope rule.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PdfSlice",
    "Unit",
    "expand_units",
    "page_texts",
    "read_slice_text",
    "slice_cache_key",
    "slice_pdf",
    "slice_unit_path",
]


@dataclass(frozen=True)
class PdfSlice:
    """A contiguous, inclusive page range of a single PDF.

    `page_start`/`page_end` are 0-indexed and inclusive. `char_len` is the
    length of the slice's extracted text, computed at slicing time so the
    packer can cost a slice without re-opening the document.
    """

    path: Path
    page_start: int
    page_end: int
    char_len: int

    @property
    def source_file(self) -> Path:
        """The parent PDF — what extraction reports as the node's source."""
        return self.path

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.path.name}#p{self.page_start}-{self.page_end}"


Unit = Path | PdfSlice

# Page-text cache: a 992-page book is read once, not once per slice.
# Keyed by (resolved path, mtime_ns, size) so an edited file invalidates.
_PAGE_CACHE: dict[tuple[str, int, int], list[str]] = {}
_PAGE_CACHE_MAX = 8


def _cache_key_for(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()), st.st_mtime_ns, st.st_size)


def page_texts(path: Path) -> list[str]:
    """Extracted text per page, via PyMuPDF with a pypdf fallback (cached)."""
    key = _cache_key_for(path)
    hit = _PAGE_CACHE.get(key)
    if hit is not None:
        return hit

    pages: list[str] = []
    try:
        import pymupdf  # PyMuPDF >= 1.24 module name

        with pymupdf.open(str(path)) as doc:
            pages = [p.get_text() for p in doc]
    except ImportError:
        try:
            import fitz

            with fitz.open(str(path)) as doc:
                pages = [p.get_text() for p in doc]
        except ImportError:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = [(p.extract_text() or "") for p in reader.pages]

    if len(_PAGE_CACHE) >= _PAGE_CACHE_MAX:
        _PAGE_CACHE.pop(next(iter(_PAGE_CACHE)))
    _PAGE_CACHE[key] = pages
    return pages


def slice_pdf(path: Path, char_cap: int) -> list[PdfSlice]:
    """Split `path` into contiguous page-range slices of at most `char_cap` chars.

    Boundary preference: never split a page. A single page whose text already
    exceeds the cap becomes its own slice and is allowed to overflow — there is
    no char-offset model for PDF text that would keep `source_location`
    meaningful, and the model's context window is far larger than the cap.
    """
    if char_cap <= 0:
        raise ValueError(f"char_cap must be positive, got {char_cap}")

    pages = page_texts(path)
    if not pages:
        return []

    slices: list[PdfSlice] = []
    start = 0
    acc = 0
    for i, text in enumerate(pages):
        n = len(text)
        # Close the current slice before adding a page that would overflow it.
        if i > start and acc + n > char_cap:
            slices.append(PdfSlice(path, start, i - 1, acc))
            start, acc = i, 0
        acc += n
    slices.append(PdfSlice(path, start, len(pages) - 1, acc))
    return slices


def read_slice_text(unit: Unit) -> str:
    """Text of a unit: whole-file for a Path, page range for a PdfSlice."""
    if isinstance(unit, PdfSlice):
        return "".join(page_texts(unit.path)[unit.page_start : unit.page_end + 1])
    try:
        return unit.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def slice_unit_path(unit: Unit) -> Path:
    """The filesystem path a unit belongs to (the parent PDF for a slice)."""
    return unit.path if isinstance(unit, PdfSlice) else unit


def slice_cache_key(unit: Unit) -> str:
    """Stable content-hash cache key for a unit.

    Hashing the slice text (not the page numbers) means re-slicing with a
    different cap reuses cached extraction for identical content, and an edited
    page invalidates only the slices that contain it.
    """
    text = read_slice_text(unit)
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    if isinstance(unit, PdfSlice):
        return f"{unit.path.name}:p{unit.page_start}-{unit.page_end}:{h[:32]}"
    return f"{unit.name}:{h[:32]}"


def expand_units(paths: list[Path], char_cap: int) -> list[Unit]:
    """Replace oversized PDFs with page-range slices; pass everything else through.

    This is the single integration point the extraction path needs: give it the
    file list it already has, get back a unit list the packer understands.
    """
    units: list[Unit] = []
    for p in paths:
        if p.suffix.lower() != ".pdf":
            units.append(p)
            continue
        try:
            sliced = slice_pdf(p, char_cap)
        except Exception:  # corrupt/unreadable PDF: fall back to upstream behaviour
            units.append(p)
            continue
        if len(sliced) <= 1:
            # Small PDF: keep the plain Path so upstream semantics are identical.
            units.append(p)
        else:
            units.extend(sliced)
    return units


def env_char_cap(default: int) -> int:
    """Slice cap, overridable via GRAPHIFY_ENT_CHAR_CAP (all config via env)."""
    raw = os.environ.get("GRAPHIFY_ENT_CHAR_CAP")
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default
