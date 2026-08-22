"""Phase 1.2 — OCR fallback for scanned PDFs (TDD: written before the implementation).

Acceptance (execution plan §1.2): scanned PDFs produce > 0 nodes each; OCR
result cached (second run: 0 OCR calls). Portable-stack amendment (ADR-0001):
Tesseract is primary, not Textract.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF required")

from graphify_ent.ocr import (  # noqa: E402
    OCR_METHOD_TAG,
    chars_per_page,
    needs_ocr,
    ocr_pdf_pages,
    ocr_cache_stats,
    reset_ocr_cache_stats,
)

HAS_TESSERACT = shutil.which("tesseract") is not None
requires_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract not installed")


def _text_pdf(path: Path, pages: int = 2) -> Path:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(36, 36, 560, 780),
            f"PAGE-{i} " + ("real embedded text " * 40),
            fontsize=11,
        )
    doc.save(str(path))
    doc.close()
    return path


def _scanned_pdf(path: Path, pages: int = 2, words: tuple[str, ...] = ("BECHAMEL", "ROUX")) -> Path:
    """A PDF whose pages carry rasterized words only — no text layer at all."""
    src = fitz.open()
    for i in range(pages):
        page = src.new_page()
        page.insert_text((72, 200), words[i % len(words)], fontsize=48)
    # Rasterize each page and rebuild as image-only pages: this is what a
    # scanner produces, and what `extract_pdf_text` returns "" for.
    out = fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=200)
        opage = out.new_page(width=page.rect.width, height=page.rect.height)
        opage.insert_image(page.rect, pixmap=pix)
    out.save(str(path))
    out.close()
    src.close()
    with fitz.open(str(path)) as check:
        assert all(len(p.get_text().strip()) == 0 for p in check), "fixture must have no text layer"
    return path


class TestTrigger:
    def test_text_pdf_does_not_need_ocr(self, tmp_path):
        pdf = _text_pdf(tmp_path / "text.pdf")
        assert chars_per_page(pdf) > 200
        assert needs_ocr(pdf) is False

    def test_scanned_pdf_needs_ocr(self, tmp_path):
        pdf = _scanned_pdf(tmp_path / "scan.pdf")
        assert chars_per_page(pdf) < 200
        assert needs_ocr(pdf) is True

    def test_threshold_is_configurable(self, tmp_path):
        pdf = _text_pdf(tmp_path / "text.pdf")
        # An absurdly high threshold forces OCR even on a text PDF.
        assert needs_ocr(pdf, min_chars_per_page=10**9) is True


@requires_tesseract
class TestOcrExtraction:
    def test_scanned_pdf_yields_text(self, tmp_path):
        pdf = _scanned_pdf(tmp_path / "scan.pdf")
        pages = ocr_pdf_pages(pdf, cache_dir=tmp_path / "cache")
        assert len(pages) == 2
        joined = " ".join(pages).upper()
        # Tesseract on clean 200-dpi renders of block capitals should recover these.
        assert "BECHAMEL" in joined or "ROUX" in joined
        assert sum(len(p) for p in pages) > 0

    def test_result_is_cached_second_run_makes_zero_ocr_calls(self, tmp_path):
        """Acceptance: OCR is the expensive step — never repeat it."""
        pdf = _scanned_pdf(tmp_path / "scan.pdf")
        cache = tmp_path / "cache"

        reset_ocr_cache_stats()
        first = ocr_pdf_pages(pdf, cache_dir=cache)
        stats_first = ocr_cache_stats()
        assert stats_first["ocr_calls"] > 0, "first run must actually OCR"

        reset_ocr_cache_stats()
        second = ocr_pdf_pages(pdf, cache_dir=cache)
        stats_second = ocr_cache_stats()
        assert second == first, "cached result must match"
        assert stats_second["ocr_calls"] == 0, "second run must make zero OCR calls"
        assert stats_second["cache_hits"] > 0

    def test_cache_is_keyed_by_content_not_path(self, tmp_path):
        pdf_a = _scanned_pdf(tmp_path / "a.pdf")
        pdf_b = _scanned_pdf(tmp_path / "b.pdf")  # identical content, different name
        cache = tmp_path / "cache"
        ocr_pdf_pages(pdf_a, cache_dir=cache)
        reset_ocr_cache_stats()
        ocr_pdf_pages(pdf_b, cache_dir=cache)
        assert ocr_cache_stats()["ocr_calls"] == 0, "identical content must hit the cache"


class TestTagging:
    def test_ocr_method_tag_is_declared(self):
        """Nodes from OCR must be down-weightable in retrieval ranking."""
        assert OCR_METHOD_TAG == "ocr"

    @requires_tesseract
    def test_extract_reports_method(self, tmp_path):
        from graphify_ent.ocr import extract_pdf_text_with_ocr

        scanned = _scanned_pdf(tmp_path / "scan.pdf")
        text, method = extract_pdf_text_with_ocr(scanned, cache_dir=tmp_path / "c")
        assert method == OCR_METHOD_TAG
        assert text.strip()

    def test_text_pdf_reports_native_method(self, tmp_path):
        from graphify_ent.ocr import extract_pdf_text_with_ocr

        pdf = _text_pdf(tmp_path / "text.pdf")
        text, method = extract_pdf_text_with_ocr(pdf, cache_dir=tmp_path / "c")
        assert method == "native"
        assert "real embedded text" in text


class TestGracefulDegradation:
    def test_missing_tesseract_returns_empty_not_crash(self, tmp_path, monkeypatch):
        import graphify_ent.ocr as ocr_mod

        pdf = _scanned_pdf(tmp_path / "scan.pdf")
        monkeypatch.setattr(ocr_mod, "_tesseract_available", lambda: False)
        pages = ocr_pdf_pages(pdf, cache_dir=tmp_path / "cache")
        assert pages == [] or all(p == "" for p in pages)
