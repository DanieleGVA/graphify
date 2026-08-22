"""Phase 1.4 — document date + authority metadata (prerequisite for Phase 6).

Acceptance (execution plan §1.4): ≥80 % of dated documents get a correct
`doc_date`; every node carries `source_rank`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphify_ent.docmeta import (
    DocMeta,
    doc_date_from_filename,
    doc_date_from_text,
    extract_doc_meta,
    load_authority,
    source_rank_for,
    version_marker,
)

fitz = pytest.importorskip("fitz", reason="PyMuPDF required")


AUTHORITY_YAML = """
default_rank: 5
domains:
  pilot:
    rules:
      - pattern: "contracts/**"
        rank: 1
      - pattern: "policies/**"
        rank: 2
      - pattern: "decks/**"
        rank: 3
"""


@pytest.fixture
def authority(tmp_path) -> Path:
    p = tmp_path / "authority.yaml"
    p.write_text(AUTHORITY_YAML)
    return p


class TestFilenameDates:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("report_2026-03-14.pdf", "2026-03-14"),
            ("Calcmenu_RecipeExport_260702_Summarized_07.20.2026.xlsx", "2026-07-20"),
            ("policy 2024_06.pdf", "2024-06-01"),
            ("minutes-20250131.pdf", "2025-01-31"),
            ("guide_v2.pdf", None),
            ("no-date-here.pdf", None),
        ],
    )
    def test_filename_patterns(self, name, expected):
        got = doc_date_from_filename(Path(name))
        assert (got.isoformat() if got else None) == expected


class TestVersionMarkers:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("policy_v2.pdf", 2),
            ("policy_v10.pdf", 10),
            ("handbook rev.3.pdf", 3),
            ("spec-V4-final.pdf", 4),
            ("plain.pdf", None),
        ],
    )
    def test_version_marker(self, name, expected):
        assert version_marker(Path(name)) == expected

    def test_version_ordering_is_numeric_not_lexical(self):
        assert version_marker(Path("p_v10.pdf")) > version_marker(Path("p_v9.pdf"))


class TestContentDates:
    def test_effective_date_in_text(self):
        text = "SUPPLIER POLICY\nEffective date: 12 March 2026\nAll kitchens must..."
        d = doc_date_from_text(text)
        assert d and d.isoformat() == "2026-03-12"

    def test_iso_date_in_text(self):
        d = doc_date_from_text("Revision approved 2025-11-02 by the board")
        assert d and d.isoformat() == "2025-11-02"

    def test_no_date_returns_none(self):
        assert doc_date_from_text("This document has no date at all.") is None

    def test_first_plausible_date_wins(self):
        text = "Effective date: 2026-01-05\nPrinted 2026-09-09"
        d = doc_date_from_text(text)
        assert d.isoformat() == "2026-01-05"


class TestAuthority:
    def test_pattern_ranking(self, authority):
        cfg = load_authority(authority)
        assert source_rank_for(Path("contracts/msc/deal.pdf"), cfg, domain="pilot") == 1
        assert source_rank_for(Path("policies/hygiene.pdf"), cfg, domain="pilot") == 2
        assert source_rank_for(Path("decks/q3.pdf"), cfg, domain="pilot") == 3

    def test_unmatched_path_gets_default_rank(self, authority):
        cfg = load_authority(authority)
        assert source_rank_for(Path("misc/notes.pdf"), cfg, domain="pilot") == 5

    def test_unknown_domain_gets_default_rank(self, authority):
        cfg = load_authority(authority)
        assert source_rank_for(Path("contracts/x.pdf"), cfg, domain="other") == 5

    def test_missing_config_is_domain_neutral_default(self, tmp_path):
        cfg = load_authority(tmp_path / "absent.yaml")
        assert source_rank_for(Path("anything.pdf"), cfg, domain="pilot") == 5


class TestExtractDocMeta:
    def _pdf(self, path: Path, text: str, metadata: dict | None = None) -> Path:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(36, 36, 560, 780), text, fontsize=10)
        if metadata:
            doc.set_metadata(metadata)
        doc.save(str(path))
        doc.close()
        return path

    def test_every_document_gets_a_source_rank(self, tmp_path, authority):
        pdf = self._pdf(tmp_path / "plain.pdf", "no dates here")
        meta = extract_doc_meta(pdf, authority_path=authority, domain="pilot")
        assert isinstance(meta, DocMeta)
        assert meta.source_rank == 5, "source_rank is mandatory on every node"

    def test_content_date_is_found_and_confident(self, tmp_path, authority):
        pdf = self._pdf(tmp_path / "policy.pdf", "Effective date: 4 May 2026\nbody text")
        meta = extract_doc_meta(pdf, authority_path=authority, domain="pilot")
        assert meta.doc_date and meta.doc_date.isoformat() == "2026-05-04"
        assert meta.doc_date_confidence >= 0.8

    def test_filename_date_is_lower_confidence_than_content(self, tmp_path, authority):
        pdf = self._pdf(tmp_path / "notes_2026-02-02.pdf", "body without a date")
        meta = extract_doc_meta(pdf, authority_path=authority, domain="pilot")
        assert meta.doc_date.isoformat() == "2026-02-02"
        assert 0 < meta.doc_date_confidence < 0.8

    def test_pdf_metadata_date_is_used_when_no_other_signal(self, tmp_path, authority):
        pdf = self._pdf(
            tmp_path / "meta.pdf", "no date in body", metadata={"creationDate": "D:20240815120000"}
        )
        meta = extract_doc_meta(pdf, authority_path=authority, domain="pilot")
        assert meta.doc_date and meta.doc_date.year == 2024

    def test_undated_document_reports_zero_confidence(self, tmp_path, authority):
        pdf = self._pdf(tmp_path / "undated.pdf", "nothing temporal")
        meta = extract_doc_meta(pdf, authority_path=authority, domain="pilot")
        assert meta.doc_date is None
        assert meta.doc_date_confidence == 0.0

    def test_version_is_captured_for_phase6(self, tmp_path, authority):
        pdf = self._pdf(tmp_path / "policy_v2.pdf", "second edition")
        meta = extract_doc_meta(pdf, authority_path=authority, domain="pilot")
        assert meta.version == 2

    def test_as_node_props_shape(self, tmp_path, authority):
        pdf = self._pdf(tmp_path / "policy_v2.pdf", "Effective date: 2026-03-01")
        props = extract_doc_meta(pdf, authority_path=authority, domain="pilot").as_node_props()
        assert props["source_rank"] == 5
        assert props["doc_date"] == "2026-03-01"
        assert props["doc_date_confidence"] > 0
        assert props["version"] == 2
