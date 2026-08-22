"""Phase 1.1 — PdfSlice page-based slicing (TDD: written before the implementation).

Acceptance (execution plan §1.1): a synthetic 300-page PDF is fully dispatched
(sum of slice chars ≈ full text chars, 0 % dropped); node `source_file` still
points at the parent PDF; cache keys per-slice are stable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF required for PDF slicing")

from graphify_ent.file_slice import (  # noqa: E402
    PdfSlice,
    read_slice_text,
    slice_cache_key,
    slice_unit_path,
    slice_pdf,
)


def _make_pdf(path: Path, pages: int, chars_per_page: int = 1200) -> Path:
    """Build a synthetic PDF with predictable, page-identifiable text.

    `insert_textbox` silently inserts *nothing* when the text overflows the
    box, so the fixture asserts the pages really carry text — otherwise every
    slicing assertion below would pass vacuously against an empty document.
    """
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        # Marker line makes it possible to assert which pages a slice covers.
        body = f"PAGE-{i:04d} " + ("lorem ipsum dolor sit amet " * (chars_per_page // 27))
        page.insert_textbox(fitz.Rect(36, 36, 560, 780), body, fontsize=8)
    doc.save(str(path))
    doc.close()
    with fitz.open(str(path)) as check:
        assert all(len(p.get_text()) > 0 for p in check), "fixture produced empty pages"
    return path


@pytest.fixture(scope="module")
def big_pdf(tmp_path_factory) -> Path:
    return _make_pdf(tmp_path_factory.mktemp("pdf") / "big.pdf", pages=300)


@pytest.fixture(scope="module")
def small_pdf(tmp_path_factory) -> Path:
    return _make_pdf(tmp_path_factory.mktemp("pdf") / "small.pdf", pages=3)


def _full_text(path: Path) -> str:
    with fitz.open(str(path)) as doc:
        return "".join(p.get_text() for p in doc)


class TestSlicing:
    def test_small_pdf_is_a_single_slice(self, small_pdf):
        slices = slice_pdf(small_pdf, char_cap=20_000)
        assert len(slices) == 1
        assert slices[0].page_start == 0
        assert slices[0].page_end == 2

    def test_no_characters_dropped_on_300_page_pdf(self, big_pdf):
        """The Phase 1.1 acceptance criterion: 0 % dropped."""
        slices = slice_pdf(big_pdf, char_cap=20_000)
        assert len(slices) > 1, "a 300-page PDF must split into multiple slices"
        dispatched = sum(len(read_slice_text(s)) for s in slices)
        full = len(_full_text(big_pdf))
        assert dispatched >= full * 0.999, f"dropped {full - dispatched} of {full} chars"

    def test_slices_are_contiguous_and_cover_every_page(self, big_pdf):
        slices = slice_pdf(big_pdf, char_cap=20_000)
        assert slices[0].page_start == 0
        with fitz.open(str(big_pdf)) as doc:
            last_page = doc.page_count - 1
        assert slices[-1].page_end == last_page
        for prev, nxt in zip(slices, slices[1:]):
            assert nxt.page_start == prev.page_end + 1, "page ranges must not gap or overlap"

    def test_each_slice_respects_the_char_cap(self, big_pdf):
        cap = 20_000
        for s in slice_pdf(big_pdf, char_cap=cap):
            text = read_slice_text(s)
            # A single page larger than the cap is allowed to overflow (it cannot
            # be split further without a char-offset model), but multi-page
            # slices must fit.
            if s.page_end > s.page_start:
                assert len(text) <= cap, f"{s} produced {len(text)} chars > cap"

    def test_oversized_single_page_gets_its_own_slice(self, tmp_path):
        """A page whose own text exceeds the cap cannot be split further.

        It becomes a one-page slice and is allowed to overflow — never dropped.
        """
        pdf = _make_pdf(tmp_path / "dense.pdf", pages=4, chars_per_page=1200)
        page_chars = min(len(t) for t in __import__("graphify_ent.file_slice",
                                                    fromlist=["page_texts"]).page_texts(pdf))
        slices = slice_pdf(pdf, char_cap=page_chars // 2)
        assert all(s.page_start == s.page_end for s in slices), "each page must stand alone"
        assert len(slices) == 4
        assert sum(len(read_slice_text(s)) for s in slices) == sum(
            len(t) for t in __import__("graphify_ent.file_slice",
                                       fromlist=["page_texts"]).page_texts(pdf)
        ), "overflowing pages must still be dispatched in full"


class TestParentReporting:
    def test_source_file_points_at_the_parent_pdf(self, big_pdf):
        for s in slice_pdf(big_pdf, char_cap=20_000):
            assert s.path == big_pdf
            assert slice_unit_path(s) == big_pdf
            assert s.source_file == big_pdf

    def test_plain_paths_pass_through_unit_helpers(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("hello world")
        assert slice_unit_path(txt) == txt
        assert read_slice_text(txt) == "hello world"


class TestCacheKeys:
    def test_cache_key_is_stable_across_calls(self, big_pdf):
        a = [slice_cache_key(s) for s in slice_pdf(big_pdf, char_cap=20_000)]
        b = [slice_cache_key(s) for s in slice_pdf(big_pdf, char_cap=20_000)]
        assert a == b

    def test_cache_key_is_unique_per_slice(self, big_pdf):
        keys = [slice_cache_key(s) for s in slice_pdf(big_pdf, char_cap=20_000)]
        assert len(keys) == len(set(keys))

    def test_cache_key_changes_when_content_changes(self, tmp_path):
        p1 = _make_pdf(tmp_path / "a.pdf", pages=2)
        p2 = _make_pdf(tmp_path / "b.pdf", pages=2, chars_per_page=2400)
        k1 = slice_cache_key(slice_pdf(p1, char_cap=20_000)[0])
        k2 = slice_cache_key(slice_pdf(p2, char_cap=20_000)[0])
        assert k1 != k2


class TestPackerIntegration:
    def test_read_files_dispatches_slices(self, big_pdf, tmp_path):
        """`graphify.llm._read_files` must treat PdfSlice like a Path."""
        from graphify.llm import _read_files

        slices = slice_pdf(big_pdf, char_cap=20_000)[:2]
        out = _read_files(list(slices), big_pdf.parent)
        assert "=== " in out
        assert len(out) > 1000
        # Both slices' text must be present.
        for s in slices:
            assert read_slice_text(s)[:80].strip()[:40] in out

    def test_estimate_file_tokens_handles_slices(self, big_pdf):
        from graphify.llm import _estimate_file_tokens

        s = slice_pdf(big_pdf, char_cap=20_000)[0]
        assert _estimate_file_tokens(s) > 0

    def test_packer_accepts_slice_units(self, big_pdf):
        from graphify.llm import _pack_chunks_by_tokens

        slices = slice_pdf(big_pdf, char_cap=20_000)
        chunks = _pack_chunks_by_tokens(list(slices), token_budget=20_000)
        assert chunks
        packed = [u for c in chunks for u in c]
        assert len(packed) == len(slices), "packing must not drop slice units"


class TestExpandUnits:
    def test_expand_units_slices_only_oversized_pdfs(self, big_pdf, small_pdf, tmp_path):
        from graphify_ent.file_slice import expand_units

        txt = tmp_path / "plain.txt"
        txt.write_text("short")
        units = expand_units([big_pdf, small_pdf, txt], char_cap=20_000)
        parents = {slice_unit_path(u) for u in units}
        assert parents == {big_pdf, small_pdf, txt}
        big_units = [u for u in units if slice_unit_path(u) == big_pdf]
        assert len(big_units) > 1
        assert any(isinstance(u, Path) for u in units)  # txt passes through
