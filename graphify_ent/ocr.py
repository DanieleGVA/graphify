"""Phase 1.2 — OCR fallback for scanned PDFs.

Upstream `detect.extract_pdf_text` uses pypdf only, so an image-only (scanned)
PDF yields "" and therefore zero nodes. This module adds the fallback:

    chars_per_page < 200  →  rasterize at 300 dpi (PyMuPDF)  →  Tesseract

Portable-stack amendment (ADR-0001): Tesseract is primary; AWS Textract is not
used anywhere. PaddleOCR remains an optional future backend behind the same
interface.

OCR is the expensive step, so results are cached on disk by **content hash** of
the rendered page image — identical content never gets OCR'd twice, regardless
of file name or path. Nodes extracted from OCR text are tagged
`extraction_method: "ocr"` so retrieval can down-weight them (Phase 3.2) and
the clash resolver can rank them below native text (Phase 6.5).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

__all__ = [
    "OCR_METHOD_TAG",
    "NATIVE_METHOD_TAG",
    "chars_per_page",
    "extract_pdf_text_with_ocr",
    "needs_ocr",
    "ocr_cache_stats",
    "ocr_pdf_pages",
    "reset_ocr_cache_stats",
]

OCR_METHOD_TAG = "ocr"
NATIVE_METHOD_TAG = "native"

#: A page with fewer characters than this is treated as scanned/near-empty.
DEFAULT_MIN_CHARS_PER_PAGE = 200
#: Rasterization resolution for OCR input (execution plan §1.2).
DEFAULT_DPI = 300

_STATS = {"ocr_calls": 0, "cache_hits": 0}


def reset_ocr_cache_stats() -> None:
    _STATS["ocr_calls"] = 0
    _STATS["cache_hits"] = 0


def ocr_cache_stats() -> dict[str, int]:
    return dict(_STATS)


def _tesseract_available() -> bool:
    """Separate function so tests can simulate a host without Tesseract."""
    return shutil.which("tesseract") is not None


def _open(path: Path):
    try:
        import pymupdf

        return pymupdf.open(str(path))
    except ImportError:
        import fitz

        return fitz.open(str(path))


def chars_per_page(path: Path) -> float:
    """Mean characters of the native text layer per page (0.0 for a scan)."""
    from graphify_ent.file_slice import page_texts

    pages = page_texts(path)
    if not pages:
        return 0.0
    return sum(len(t) for t in pages) / len(pages)


def needs_ocr(path: Path, min_chars_per_page: int = DEFAULT_MIN_CHARS_PER_PAGE) -> bool:
    """True when the native text layer is too sparse to be the real content."""
    return chars_per_page(path) < min_chars_per_page


def _default_cache_dir() -> Path:
    """Cache location; all config via env per CLAUDE.md."""
    return Path(os.environ.get("GRAPHIFY_ENT_OCR_CACHE", ".graphify-ent/ocr-cache"))


def _page_image_bytes(doc, index: int, dpi: int) -> bytes:
    pix = doc[index].get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def _cache_path(cache_dir: Path, digest: str) -> Path:
    # Two-level fan-out keeps directories small on a 10 GB corpus.
    return cache_dir / digest[:2] / f"{digest}.txt"


def ocr_pdf_pages(
    path: Path,
    cache_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
    lang: str | None = None,
) -> list[str]:
    """OCR every page of `path`, returning one text string per page.

    Returns `[]` when Tesseract is unavailable — a missing OCR backend degrades
    to "no extra text", never to a crash mid-corpus.
    """
    if not _tesseract_available():
        return []

    import pytesseract

    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    lang = lang or os.environ.get("GRAPHIFY_ENT_OCR_LANG", "eng")

    out: list[str] = []
    with _open(path) as doc:
        for i in range(doc.page_count):
            png = _page_image_bytes(doc, i, dpi)
            digest = hashlib.sha256(png + lang.encode()).hexdigest()
            cached = _cache_path(cache_dir, digest)
            if cached.exists():
                _STATS["cache_hits"] += 1
                out.append(cached.read_text(encoding="utf-8"))
                continue

            from io import BytesIO

            from PIL import Image

            text = pytesseract.image_to_string(Image.open(BytesIO(png)), lang=lang)
            _STATS["ocr_calls"] += 1
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(text, encoding="utf-8")
            out.append(text)
    return out


def extract_pdf_text_with_ocr(
    path: Path,
    cache_dir: Path | None = None,
    min_chars_per_page: int = DEFAULT_MIN_CHARS_PER_PAGE,
    dpi: int = DEFAULT_DPI,
) -> tuple[str, str]:
    """Return (text, method) where method is "native" or "ocr".

    Native extraction wins whenever the text layer is dense enough; OCR is a
    fallback, never a replacement.
    """
    from graphify_ent.file_slice import page_texts

    if not needs_ocr(path, min_chars_per_page):
        return "".join(page_texts(path)), NATIVE_METHOD_TAG

    ocr_pages = ocr_pdf_pages(path, cache_dir=cache_dir, dpi=dpi)
    if not ocr_pages or not "".join(ocr_pages).strip():
        # OCR unavailable or produced nothing: fall back to whatever native gave.
        return "".join(page_texts(path)), NATIVE_METHOD_TAG
    return "".join(ocr_pages), OCR_METHOD_TAG


def ocr_page_texts_for_slicing(
    path: Path, cache_dir: Path | None = None, dpi: int = DEFAULT_DPI
) -> list[str] | None:
    """Per-page OCR text for a scanned PDF, shaped for `file_slice.slice_pdf`.

    Returns None when the document does not need OCR, so callers can keep the
    native page-text path untouched.
    """
    if not needs_ocr(path):
        return None
    pages = ocr_pdf_pages(path, cache_dir=cache_dir, dpi=dpi)
    return pages or None
