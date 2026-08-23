#!/usr/bin/env python3
"""Build the passage layer: one node per page, carrying that page's text.

The graph needs two kinds of node and they answer different questions.

  concept nodes  — what the document talks about, and how those things relate.
                   Produced by the LLM, one claim each, cheap to compare.
  passage nodes  — what the document actually SAYS, page by page. Produced
                   here, deterministically, with no model involved.

The recipe-verification case is what forced the distinction. Asked for the roux
proportions, concept nodes returned `'clarified butter'` and `'sifted flour'`
and dropped the 300 g and 350 g — they know WHERE, never WHAT. Settling a claim
against a source needs the sentence, so a layer has to carry the sentence.

Deterministic on purpose: page text is not something to pay a model to guess at,
and a passage that came from an extraction step could be wrong in ways the
verification case cannot tolerate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz

MIN_CHARS = 120          # blank pages and plates carry nothing to retrieve
EVIDENCE_CHARS = 300


def slug(s: str, n: int = 28) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower())[:n].strip("_")


def build(pdf: Path, domain: str) -> list[dict]:
    doc = fitz.open(pdf)
    nodes: list[dict] = []
    stem = slug(pdf.stem)
    for i in range(doc.page_count):
        text = doc[i].get_text()
        if len(text.strip()) < MIN_CHARS:
            continue
        page = i + 1
        # A page's own first lines are its evidence: literally present in the
        # source, which is what the Q1 gate checks.
        evidence = re.sub(r"\s+", " ", text).strip()[:EVIDENCE_CHARS]
        heading = next((ln.strip() for ln in text.split("\n")
                        if 6 <= len(ln.strip()) <= 70), f"page {page}")
        nodes.append({
            "id": f"{stem}_p{page:04d}",
            "label": heading[:80],
            "label_orig": heading[:80],
            "lang": "en",
            "file_type": "entity",
            "source_file": pdf.name,
            "source_location": f"pages {page}-{page}",
            "page_lo": page, "page_hi": page,
            "evidence": evidence,
            "text_excerpt": evidence,
            "passage": text.strip(),
            "confidence": "EXTRACTED",
            "extraction_method": "page",
            "domain": domain,
        })
    doc.close()
    return nodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--domain", default="pilot")
    ap.add_argument("--out", type=Path, default=Path("../evidence/T73/pages.json"))
    args = ap.parse_args()

    nodes = build(args.pdf, args.domain)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"nodes": nodes, "edges": []}, ensure_ascii=False))
    chars = sum(len(n["passage"]) for n in nodes)
    print(f"{len(nodes)} pagine con contenuto, {chars:,} caratteri "
          f"(media {chars // max(len(nodes), 1):,} per pagina) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
