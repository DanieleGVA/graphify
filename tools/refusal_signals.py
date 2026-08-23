#!/usr/bin/env python3
"""Which signal still separates "the corpus answers this" from "it does not"?

Measured on the 74,680-node graph, absolute vector similarity no longer does:
in-corpus scores run 0.737–0.923 and out-of-corpus 0.718–0.807, so the only
floor that refuses everything unanswerable (0.81) also refuses 81 of 145
answerable questions. That is not a floor, it is a mute button.

The reason is structural, not a bug. Similarity is a distance to the *nearest*
node, and a graph 23× denser has a nearer node for everything. So test signals
that are about the SHAPE of the neighbourhood rather than its distance:

  best            the current signal, kept as the baseline
  gap             best − mean(the rest of the top-k). A question the corpus
                  answers should have one clear winner; a question it does not
                  answer sits in a flat field of equally mediocre neighbours.
  ratio           best ÷ mean(the rest), the scale-free form of the same idea
  lexical         does the best hit's own text share a word with the query
  gap_x_best      the two independent signals combined

Each is swept the same way and reported with both costs, so the choice stays a
policy decision made on visible numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from calibrate_refusal import OUT_OF_CORPUS

from graphify_ent.embed import Embedder
from graphify_ent.loader import Neo4jLoader
from graphify_ent.retrieval import HybridRetriever

_TOK = re.compile(r"[a-zà-ÿ]{4,}")


def features(retriever: HybridRetriever, embedder: Embedder, q: str,
             domain: str = "pilot") -> dict:
    vec = retriever.vector_search(embedder.encode([q])[0], top_k=25, domain=domain)
    if not vec:
        return {"best": 0.0, "gap": 0.0, "ratio": 0.0, "lexical": 0.0}
    scores = [s for _, s in vec]
    best = scores[0]
    rest = scores[1:] or [best]
    mean_rest = statistics.mean(rest)
    props = retriever.hydrate([i for i, _ in vec[:5]])
    qt = set(_TOK.findall(q.lower()))
    lex = 0.0
    for p in props.values():
        text = f"{p.get('label','')} {p.get('text_excerpt','')} {p.get('evidence','')}".lower()
        if qt & set(_TOK.findall(text)):
            lex = 1.0
            break
    return {"best": best, "gap": best - mean_rest,
            "ratio": best / mean_rest if mean_rest else 0.0, "lexical": lex}


def sweep(name: str, pos: list[float], neg: list[float], lo: float, hi: float,
          step: float) -> dict:
    rows = []
    x = lo
    while x <= hi + 1e-9:
        rows.append({"floor": round(x, 4),
                     "refused": sum(1 for s in neg if s < x),
                     "lost": sum(1 for s in pos if s < x)})
        x += step
    full = [r for r in rows if r["refused"] == len(neg)]
    best = min(full, key=lambda r: r["lost"]) if full else None
    # the most useful operating point even when perfect separation is absent:
    # maximise refusals while losing at most 10% of answerable questions
    budget = max(1, int(0.10 * len(pos)))
    practical = max([r for r in rows if r["lost"] <= budget],
                    key=lambda r: (r["refused"], -r["lost"]), default=None)
    return {"signal": name, "separable": bool(full),
            "perfect": best, "practical": practical,
            "pos_band": [round(min(pos), 4), round(statistics.median(pos), 4),
                         round(max(pos), 4)],
            "neg_band": [round(min(neg), 4), round(statistics.median(neg), 4),
                         round(max(neg), 4)],
            "n_pos": len(pos), "n_neg": len(neg)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerable", type=Path, default=Path("../eval/neutral-qa-v1.json"))
    ap.add_argument("--golden", type=Path, default=Path("../eval/golden-qa-v1.json"))
    ap.add_argument("--json", type=Path,
                    default=Path("../evidence/T71/refusal-signals.json"))
    args = ap.parse_args()

    pos_q = [p["query"] for p in json.loads(args.answerable.read_text())["pairs"]]
    neg_q = list(OUT_OF_CORPUS)
    if args.golden.exists():
        neg_q += [p["query"] for p in json.loads(args.golden.read_text())["pairs"]
                  if p.get("kind") == "unanswerable"]

    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    try:
        pos = [features(retriever, embedder, q) for q in pos_q]
        neg = [features(retriever, embedder, q) for q in neg_q]
    finally:
        loader.close()

    out = []
    for name, lo, hi, step in (("best", 0.60, 0.95, 0.005),
                               ("gap", 0.0, 0.30, 0.002),
                               ("ratio", 1.0, 1.40, 0.005)):
        out.append(sweep(name, [f[name] for f in pos], [f[name] for f in neg],
                         lo, hi, step))

    lex_pos = sum(f["lexical"] for f in pos)
    lex_neg = sum(f["lexical"] for f in neg)
    out.append({"signal": "lexical", "separable": lex_neg == 0,
                "pos_with_overlap": f"{int(lex_pos)}/{len(pos)}",
                "neg_with_overlap": f"{int(lex_neg)}/{len(neg)}"})

    print(f"{'segnale':<10}{'nel corpus (min/med/max)':<32}"
          f"{'fuori corpus':<30}{'esito'}")
    for r in out:
        if r["signal"] == "lexical":
            print(f"{'lexical':<10}{r['pos_with_overlap']:<32}{r['neg_with_overlap']:<30}"
                  f"{'separa' if r['separable'] else 'non separa'}")
            continue
        pb, nb = r["pos_band"], r["neg_band"]
        verdict = "SEPARA" if r["separable"] else "sovrapposte"
        print(f"{r['signal']:<10}{str(pb):<32}{str(nb):<30}{verdict}")
        if r["perfect"]:
            print(f"{'':<10}  separazione perfetta a {r['perfect']['floor']}: "
                  f"perde {r['perfect']['lost']}/{r['n_pos']}")
        if r["practical"]:
            p = r["practical"]
            print(f"{'':<10}  punto pratico {p['floor']}: rifiuta {p['refused']}/{r['n_neg']}"
                  f", perde {p['lost']}/{r['n_pos']}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"signals": out,
         "raw": {"pos": pos, "neg": neg, "neg_queries": neg_q}},
        indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
