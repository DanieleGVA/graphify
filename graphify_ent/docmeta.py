"""Phase 1.4 — document date & authority metadata.

Prerequisite for Phase 6: the bitemporal model needs *valid time*, and valid
time needs a document date. This module derives, per source file:

    doc_date              when the document's content takes effect
    doc_date_confidence   how much the resolution policy may trust it
    source_rank           per-domain authority (authority.yaml, path patterns)
    version               numeric version marker (_v2, rev.3) for 6.2

Signal precedence (highest confidence first): an explicit effective/signature
date in the content, a date in the filename, then the PDF/DOCX metadata
creation date — file metadata is the weakest signal because it records when the
file was *written*, not when the content took effect.

Domain-agnostic: `authority.yaml` is a per-corpus path-pattern → rank map. No
domain semantics are hardcoded here (CLAUDE.md DOMAIN-AGNOSTIC rule).
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DocMeta",
    "doc_date_from_filename",
    "doc_date_from_metadata",
    "doc_date_from_text",
    "extract_doc_meta",
    "load_authority",
    "source_rank_for",
    "version_marker",
]

DEFAULT_RANK = 5

CONFIDENCE_CONTENT = 0.9
CONFIDENCE_FILENAME = 0.6
CONFIDENCE_METADATA = 0.4
CONFIDENCE_NONE = 0.0

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
_MONTHS.update({m[:3].lower(): i for m, i in list(_MONTHS.items())})
# Italian and French month names — the corpus is IT/EN/FR by design.
_MONTHS.update({
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6,
    "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10,
    "novembre": 11, "décembre": 12, "decembre": 12,
})

_EFFECTIVE_MARKERS = (
    r"effective\s+(?:date|from|as\s+of)",
    r"valid\s+from",
    r"in\s+vigore\s+dal",
    r"data\s+di\s+decorrenza",
    r"en\s+vigueur\s+(?:le|à\s+compter\s+du)",
    r"date\s+d['’]effet",
    r"signed\s+on",
    r"revision\s+approved",
    r"approved\s+on",
)


def _safe_date(y: int, m: int, d: int) -> _dt.date | None:
    try:
        date = _dt.date(y, m, d)
    except ValueError:
        return None
    # Reject implausible extremes; a corpus date outside this band is noise.
    return date if 1900 <= date.year <= 2100 else None


def doc_date_from_filename(path: Path) -> _dt.date | None:
    """Date encoded in the filename, or None.

    Recognises ISO (2026-03-14), compact (20250131), US dotted (07.20.2026),
    and year-month (2024_06 → first of month).
    """
    name = path.stem

    m = re.search(r"(20\d{2}|19\d{2})[-_.](\d{1,2})[-_.](\d{1,2})", name)
    if m and (d := _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))):
        return d

    m = re.search(r"(\d{1,2})[-_.](\d{1,2})[-_.](20\d{2}|19\d{2})", name)
    if m and (d := _safe_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))):
        return d

    m = re.search(r"(?<!\d)(20\d{2}|19\d{2})(\d{2})(\d{2})(?!\d)", name)
    if m and (d := _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))):
        return d

    m = re.search(r"(?<!\d)(20\d{2}|19\d{2})[-_.](\d{1,2})(?!\d)", name)
    if m and (d := _safe_date(int(m.group(1)), int(m.group(2)), 1)):
        return d

    return None


def _parse_textual_date(chunk: str) -> _dt.date | None:
    m = re.search(r"(20\d{2}|19\d{2})-(\d{1,2})-(\d{1,2})", chunk)
    if m and (d := _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))):
        return d

    names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    m = re.search(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({names})\.?\s+(20\d{{2}}|19\d{{2}})",
                  chunk, re.IGNORECASE)
    if m and (d := _safe_date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))):
        return d

    m = re.search(rf"({names})\.?\s+(\d{{1,2}}),?\s+(20\d{{2}}|19\d{{2}})", chunk, re.IGNORECASE)
    if m and (d := _safe_date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))):
        return d

    m = re.search(r"(\d{1,2})/(\d{1,2})/(20\d{2}|19\d{2})", chunk)
    if m and (d := _safe_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))):
        return d
    return None


def doc_date_from_text(text: str, window: int = 120) -> _dt.date | None:
    """Effective/signature date found in document content.

    Marker-anchored dates win; a bare date anywhere in the text is the fallback.
    Only the first plausible hit is returned — later dates in a document are
    usually print dates or references, not the effective date.
    """
    if not text:
        return None

    for marker in _EFFECTIVE_MARKERS:
        for m in re.finditer(marker, text, re.IGNORECASE):
            chunk = text[m.end() : m.end() + window]
            if d := _parse_textual_date(chunk):
                return d

    return _parse_textual_date(text[:4000])


def doc_date_from_metadata(path: Path) -> _dt.date | None:
    """Creation date from PDF metadata (weakest signal)."""
    if path.suffix.lower() != ".pdf":
        return None
    try:
        try:
            import pymupdf as _f
        except ImportError:
            import fitz as _f
        with _f.open(str(path)) as doc:
            raw = (doc.metadata or {}).get("creationDate") or ""
    except Exception:
        return None
    m = re.search(r"D:(\d{4})(\d{2})(\d{2})", raw or "")
    return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def version_marker(path: Path) -> int | None:
    """Numeric version from the filename (`_v2`, `rev.3`, `-V4-`), else None."""
    name = path.stem
    m = re.search(r"(?:^|[\s_\-.])v(?:er(?:sion)?)?[\s_\-.]?(\d{1,3})(?:$|[\s_\-.])",
                  name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"rev(?:ision)?[\s_\-.]*(\d{1,3})", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def load_authority(path: Path | None) -> dict:
    """Load authority.yaml; a missing or unreadable file yields the neutral default."""
    if not path or not Path(path).exists():
        return {"default_rank": DEFAULT_RANK, "domains": {}}
    try:
        import yaml

        cfg = yaml.safe_load(Path(path).read_text()) or {}
    except Exception:
        return {"default_rank": DEFAULT_RANK, "domains": {}}
    cfg.setdefault("default_rank", DEFAULT_RANK)
    cfg["domains"] = cfg.get("domains") or {}
    return cfg


def source_rank_for(path: Path, cfg: dict, domain: str | None = None) -> int:
    """Authority rank (1 = highest) for a path under a domain's rule set."""
    default = int(cfg.get("default_rank", DEFAULT_RANK))
    rules = ((cfg.get("domains") or {}).get(domain) or {}).get("rules") or []
    posix = Path(path).as_posix()
    for rule in rules:
        pattern = rule.get("pattern")
        if pattern and (fnmatch.fnmatch(posix, pattern) or fnmatch.fnmatch(posix, f"*/{pattern}")):
            return int(rule.get("rank", default))
    return default


@dataclass
class DocMeta:
    """Per-source-file metadata stamped onto every node at load time."""

    path: Path
    doc_date: _dt.date | None
    doc_date_confidence: float
    source_rank: int
    version: int | None = None
    doc_date_source: str = "none"

    def as_node_props(self) -> dict:
        return {
            "doc_date": self.doc_date.isoformat() if self.doc_date else None,
            "doc_date_confidence": self.doc_date_confidence,
            "source_rank": self.source_rank,
            "version": self.version,
            "doc_date_source": self.doc_date_source,
        }


def _head_and_tail_text(path: Path, chars: int = 6000) -> str:
    """First + last slice only — cheap, per the execution plan."""
    if path.suffix.lower() != ".pdf":
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:chars]
        except OSError:
            return ""
    try:
        from graphify_ent.file_slice import page_texts

        pages = page_texts(path)
    except Exception:
        return ""
    if not pages:
        return ""
    head = "".join(pages[:5])[:chars]
    tail = "".join(pages[-3:])[:chars]
    return f"{head}\n{tail}"


def extract_doc_meta(
    path: Path,
    authority_path: Path | None = None,
    domain: str | None = None,
) -> DocMeta:
    """Derive date/authority metadata for one source file."""
    path = Path(path)
    cfg = load_authority(authority_path)
    rank = source_rank_for(path, cfg, domain)

    if (d := doc_date_from_text(_head_and_tail_text(path))) is not None:
        return DocMeta(path, d, CONFIDENCE_CONTENT, rank, version_marker(path), "content")
    if (d := doc_date_from_filename(path)) is not None:
        return DocMeta(path, d, CONFIDENCE_FILENAME, rank, version_marker(path), "filename")
    if (d := doc_date_from_metadata(path)) is not None:
        return DocMeta(path, d, CONFIDENCE_METADATA, rank, version_marker(path), "metadata")
    return DocMeta(path, None, CONFIDENCE_NONE, rank, version_marker(path), "none")
