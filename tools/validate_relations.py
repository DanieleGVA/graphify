#!/usr/bin/env python3
"""Repeat the relation-vocabulary test across slices, books and languages.

The original finding came from ONE slice of ONE book — and that slice was a
catalogue of consommé recipes, i.e. ingredient lists. `requires` dominating
there may say more about the passage than about the vocabulary. This repeats
the A-vs-C comparison across content types and languages so the conclusion is
not an artefact of one page.

Reports per slice and aggregated, with the spread, so a difference smaller
than the between-slice variation is not read as a real effect.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request
from collections import Counter
from pathlib import Path

from graphify_ent.compact_schema import parse_compact
from graphify_ent.file_slice import read_slice_text, slice_pdf

# A = the pre-fix vocabulary (code-analysis relations only), C = current prompt.
from graphify_ent.compact_schema import COMPACT_EXTRACTION_SYSTEM as PROMPT_C

PROMPT_A = re.sub(
    r"Relation types —.*?references               generic mention, when none of the above fits\n\n",
    'Rules: relations are one of calls, implements, references, cites,\n'
    'conceptually_related_to, shares_data_with, semantically_similar_to.\n\n',
    PROMPT_C, flags=re.S)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def run(model: str, system: str, user: str, timeout: int = 900):
    payload = json.dumps({
        "model": model, "stream": False, "think": False,
        "options": {"temperature": 0, "num_ctx": 32768, "num_predict": 16000},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        env = json.loads(r.read())
    return (env.get("message") or {}).get("content") or "", env.get("eval_count", 0)


def score(content: str, source: str) -> dict:
    nodes, edges = parse_compact(content)
    src = norm(source)
    kept = [n for n in nodes if norm(n.get("evidence", "")) and norm(n["evidence"]) in src]
    ids = {n["id"] for n in kept}
    ke = [e for e in edges if e["source"] in ids and e["target"] in ids]
    rels = Counter(e["relation"] for e in ke)
    # "Uninformative" = the generic fallback, or any single label monopolising
    # the response: both carry the same (near-zero) information.
    top_share = (rels.most_common(1)[0][1] / len(ke) * 100) if ke else 0.0
    return {
        "nodes_verified": len(kept), "nodes_returned": len(nodes),
        "verified_pct": round(100 * len(kept) / len(nodes), 1) if nodes else 0.0,
        "edges": len(ke), "types": len(rels),
        "generic_pct": round(100 * rels.get("references", 0) / len(ke), 1) if ke else 0.0,
        "dominant_label": rels.most_common(1)[0][0] if rels else None,
        "dominant_share_pct": round(top_share, 1),
        "distribution": dict(rels.most_common(5)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash:cloud")
    ap.add_argument("--slices-per-book", type=int, default=2)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    books = [
        ("Escoffier  Le Guide Culinaire - Georges Auguste Escoffier.pdf", "fr", [20, 60]),
        ("The Professional Chef - The Culinary Institute of America.pdf", "en", [20, 50]),
        ("China The Cookbook - Kei Lum Chan, Diora Fong Chan.pdf", "en", [15, 45]),
    ]
    root = Path("../pilot")
    rows = []

    for fname, lang, idxs in books:
        pdf = root / fname
        if not pdf.exists():
            print(f"manca {fname}")
            continue
        slices = slice_pdf(pdf, 20_000)
        for si in idxs[: args.slices_per_book]:
            if si >= len(slices):
                continue
            sl = slices[si]
            text = read_slice_text(sl)
            user = f"=== {pdf.name} (pages {sl.page_start+1}-{sl.page_end+1}) ===\n{text}"
            for vname, sysprompt in (("A_before", PROMPT_A), ("C_after", PROMPT_C)):
                t0 = time.perf_counter()
                try:
                    content, _ = run(args.model, sysprompt, user)
                except Exception as exc:
                    print(f"  {pdf.name[:22]} s{si} {vname}: FALLITO {str(exc)[:80]}")
                    continue
                s = score(content, text)
                s.update({"book": pdf.name[:26], "lang": lang, "slice": si,
                          "variant": vname, "wall_s": round(time.perf_counter() - t0, 1)})
                rows.append(s)
                print(f"  {pdf.name[:24]:<26} s{si:<3} {vname:<9} "
                      f"verif {s['verified_pct']:>5}%  archi {s['edges']:>3}  "
                      f"tipi {s['types']}  generico {s['generic_pct']:>5}%  "
                      f"dominante {s['dominant_label']} {s['dominant_share_pct']}%",
                      flush=True)

    print("\n" + "=" * 78)
    for v in ("A_before", "C_after"):
        sel = [r for r in rows if r["variant"] == v]
        if not sel:
            continue
        gen = [r["generic_pct"] for r in sel]
        dom = [r["dominant_share_pct"] for r in sel]
        ver = [r["verified_pct"] for r in sel]
        typ = [r["types"] for r in sel]
        sd = statistics.stdev(dom) if len(dom) > 1 else 0.0
        print(f"{v:<9} n={len(sel)}  generico {statistics.mean(gen):5.1f}%  "
              f"dominante {statistics.mean(dom):5.1f}% (sd {sd:4.1f})  "
              f"tipi {statistics.mean(typ):4.1f}  verifica {statistics.mean(ver):5.1f}%")
        labels = Counter(r["dominant_label"] for r in sel)
        print(f"          etichette dominanti: {dict(labels)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
