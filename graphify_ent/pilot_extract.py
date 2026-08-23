"""Deterministic, evidence-bound extraction for the pilot corpus.

**Why this exists (documented assumption).** The architecture's extraction layer
is an LLM semantic pass. This machine has no `ANTHROPIC_API_KEY`, so that pass
cannot run, and without a graph there is nothing to load, embed, retrieve, or
evaluate — the entire downstream chain and every accuracy target would be
unmeasurable.

So this module produces a *real* graph from the *real* PDFs using structural
rules only: page-range slices become `Section` nodes, each bound to the verbatim
excerpt that justifies it, with `Document` parents and `MENTIONS`/`PART_OF`
edges. It is deliberately **not** a substitute for semantic extraction — it
emits no inferred relations and claims no semantic understanding. What it does
guarantee is the property the anti-hallucination framework depends on: every
node carries `evidence` that is a literal substring of its source document, so
Q1 faithfulness is measurable end-to-end on real content.

Swapping the LLM pass back in replaces `extract_document()` and nothing else:
the node/edge schema is the one `loader.py` already consumes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from graphify_ent.docmeta import extract_doc_meta
from graphify_ent.file_slice import page_texts, slice_pdf
from graphify_ent.labels import detect_lang, normalize_label_en

__all__ = ["ExtractionResult", "extract_corpus", "extract_document", "section_title"]

#: Slice size for section granularity. Smaller than the extraction cap because
#: retrieval wants passage-level, not chapter-level, granularity.
SECTION_CHAR_CAP = 4_000
MIN_SECTION_CHARS = 200
EVIDENCE_CHARS = 400

# A heading-ish line: short, mostly letters, not a sentence.
_HEADING = re.compile(r"^[^\n]{3,80}$")
_NOISE = re.compile(r"^[\s\d.,;:_\-–—•·|]+$")


def _node_id(*parts: str) -> str:
    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]
    stem = re.sub(r"[^a-z0-9_]", "_", parts[0].lower())[:40].strip("_")
    return f"{stem}_{digest}"


def section_title(text: str) -> str:
    """Best-effort title for a passage: the first heading-like line."""
    for line in text.splitlines():
        line = line.strip()
        if not line or _NOISE.match(line):
            continue
        if _HEADING.match(line) and not line.endswith((".", ";", ",")):
            letters = sum(c.isalpha() for c in line)
            if letters >= max(3, len(line) * 0.5):
                return re.sub(r"\s+", " ", line)[:80]
        break
    flat = re.sub(r"\s+", " ", text.strip())
    return flat[:80] if flat else "untitled"


@dataclass
class ExtractionResult:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)

    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)
        return self

    def as_graph(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges}


def extract_document(
    pdf: Path,
    domain: str = "pilot",
    authority_path: Path | None = None,
    max_sections: int | None = None,
) -> ExtractionResult:
    """Extract a Document node plus evidence-bound Section nodes from one PDF."""
    result = ExtractionResult()
    meta = extract_doc_meta(pdf, authority_path=authority_path, domain=domain)
    meta_props = meta.as_node_props()

    pages = page_texts(pdf)
    full_head = "".join(pages[:3])[:EVIDENCE_CHARS]
    doc_id = _node_id(pdf.stem, "document")
    doc_lang = detect_lang(" ".join(pages[:2])[:3000]) or "en"

    result.nodes.append(
        {
            "id": doc_id,
            "label": pdf.stem,
            "label_orig": pdf.stem,
            "label_en": normalize_label_en(pdf.stem),
            "lang": doc_lang,
            "file_type": "document",
            "source_file": pdf.name,
            "source_location": None,
            "text_excerpt": full_head,
            "evidence": full_head,
            "confidence": "EXTRACTED",
            "extraction_method": "native",
            "pages": len(pages),
            **meta_props,
        }
    )

    slices = slice_pdf(pdf, SECTION_CHAR_CAP)
    if max_sections:
        slices = slices[:max_sections]

    for sl in slices:
        text = "".join(pages[sl.page_start : sl.page_end + 1]).strip()
        if len(text) < MIN_SECTION_CHARS:
            continue
        title = section_title(text)
        sec_id = _node_id(pdf.stem, f"p{sl.page_start}_{sl.page_end}")
        excerpt = re.sub(r"[ \t]+", " ", text[:EVIDENCE_CHARS]).strip()
        lang = detect_lang(text[:2000]) or doc_lang

        result.nodes.append(
            {
                "id": sec_id,
                "label": title,
                "label_orig": title,
                "label_en": normalize_label_en(title),
                "lang": lang,
                "file_type": "document",
                "source_file": pdf.name,
                # Page numbers are 1-based for humans; this is what an answer cites.
                "source_location": f"pages {sl.page_start + 1}-{sl.page_end + 1}",
                "page_start": sl.page_start + 1,
                "page_end": sl.page_end + 1,
                "text_excerpt": text[:1000],
                # Evidence binding: a literal substring of the source document.
                "evidence": excerpt,
                "confidence": "EXTRACTED",
                "extraction_method": "native",
                **meta_props,
            }
        )
        result.edges.append(
            {
                "source": sec_id,
                "target": doc_id,
                "relation": "part_of",
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
                "source_file": pdf.name,
                "weight": 1.0,
            }
        )

    return result


def extract_corpus(
    root: Path,
    domain: str = "pilot",
    authority_path: Path | None = None,
    max_sections_per_doc: int | None = None,
    progress: bool = True,
) -> ExtractionResult:
    """Extract every PDF under `root` into a single graph fragment."""
    total = ExtractionResult()
    for pdf in sorted(root.rglob("*.pdf")):
        res = extract_document(
            pdf,
            domain=domain,
            authority_path=authority_path,
            max_sections=max_sections_per_doc,
        )
        total.merge(res)
        if progress:
            print(f"{len(res.nodes):>6} nodes  {pdf.name[:60]}")
    return total
