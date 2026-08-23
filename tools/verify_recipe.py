#!/usr/bin/env python3
"""Check a recipe card against the corpus — through the graph, not the PDF.

Reads a card as JSON, asks the graph about each assertion on it, and prints a
verdict per claim with the passage it was decided from. The PDF is never opened:
if the graph cannot settle a claim the answer is NOT_FOUND, which is a finding
about the corpus and not a cue to go read the book.

    python tools/verify_recipe.py ../eval/cards/mornay.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from graphify_ent.embed import Embedder
from graphify_ent.loader import Neo4jLoader
from graphify_ent.retrieval import HybridRetriever
from graphify_ent.verify import CONTRADICTED, NOT_FOUND, SUPPORTED, Claim, Verifier

_MARK = {SUPPORTED: "CONFERMATA", CONTRADICTED: "SMENTITA", NOT_FOUND: "NON TROVATA"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("card", type=Path)
    ap.add_argument("--domain", default="pilot")
    ap.add_argument("--json", type=Path, default=Path("../evidence/T73/verify-report.json"))
    args = ap.parse_args()

    card = json.loads(args.card.read_text())
    claims = [Claim(**c) for c in card["claims"]]

    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    embedder.encode(["warm up"])          # model load is setup, not query cost

    verifier = Verifier(retriever, embed_fn=lambda q: embedder.encode([q])[0],
                        domain=args.domain)
    t0 = time.perf_counter()
    try:
        findings = verifier.check_all(claims)
    finally:
        loader.close()
    wall = (time.perf_counter() - t0) * 1000

    print("=" * 96)
    print(f"SCHEDA: {card.get('title', args.card.stem)}")
    print(f"riferimento dichiarato: {card.get('claimed_reference', '—')}")
    print("=" * 96)
    for f in findings:
        what = f"{f.claim.subject} — {f.claim.aspect}" if f.claim.aspect else f.claim.subject
        val = f" [{f.claim.value}]" if f.claim.value else ""
        print(f"\n{_MARK[f.verdict]:<12} {what[:56]:<58}{val}")
        if f.detail:
            print(f"   {f.detail}")
        if f.evidence:
            txt = re.sub(r"\s+", " ", f.evidence)
            print(f"   fonte: {f.source_file[:40]} {f.source_location}")
            print(f"   «{txt[:260]}»")
        print(f"   ({f.latency_ms:.0f} ms, canale {f.channel})")

    counts = {v: sum(1 for f in findings if f.verdict == v)
              for v in (SUPPORTED, CONTRADICTED, NOT_FOUND)}
    report = {
        "card": card.get("title"), "claims": len(findings), "counts": counts,
        "total_ms": round(wall, 1),
        "ms_per_claim": round(wall / max(len(findings), 1), 1),
        "used_pdf": False,
        "findings": [f.as_dict() for f in findings],
    }
    print("\n" + "-" * 96)
    print(f"{counts[SUPPORTED]} confermate · {counts[CONTRADICTED]} smentite · "
          f"{counts[NOT_FOUND]} non trovate   |   "
          f"{wall:.0f} ms totali, {wall / max(len(findings), 1):.0f} ms per affermazione")
    print("nessun PDF è stato aperto: ogni verdetto viene dal grafo.")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
