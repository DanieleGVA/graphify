#!/usr/bin/env python3
"""Re-calibrate the refusal floor for the graph as it actually is.

`MIN_VECTOR_SIMILARITY = 0.72` was measured on a 3,187-node graph. The floor is
not a property of the question, it is a property of the *neighbourhood*: with
82,845 nodes something always sits close enough, so the system stopped saying
"not present in corpus" — measured, refusals went from 3/3 to 0/3. That is the
one guarantee the project cannot trade away, so the floor is re-measured rather
than nudged.

Method, unchanged from the original calibration (evidence/T32): take the best
vector similarity for a set of questions the corpus DOES answer and for a set
it does NOT, then sweep the floor and report both costs at every step. The
floor is a policy choice made on visible numbers, never a guess.

Out-of-corpus questions are multi-word phrases from domains a cookbook corpus
cannot cover — software, medicine, law, finance, astronomy — in the corpus's
own three languages, so a refusal cannot be an artefact of language mismatch.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from graphify_ent.embed import Embedder
from graphify_ent.loader import Neo4jLoader
from graphify_ent.retrieval import HybridRetriever

OUT_OF_CORPUS = [
    # software / infrastructure
    "kubernetes ingress controller tls termination",
    "garbage collector generational heap tuning",
    "distributed consensus raft leader election",
    "certificato x509 rinnovo automatico",
    "requête sql jointure externe optimisation",
    "compilateur inférence de types génériques",
    # medicine
    "myocardial infarction troponin elevation",
    "antibiotic resistance plasmid transfer",
    "risonanza magnetica sequenza di diffusione",
    "posologie anticoagulant fibrillation auriculaire",
    # law / finance
    "employee stock option vesting schedule",
    "intellectual property licensing indemnity clause",
    "ammortamento fiscale cespiti pluriennali",
    "clause résolutoire bail commercial",
    "sovereign bond yield curve inversion",
    # astronomy / physics
    "gravitational lensing dark matter halo",
    "semiconductor band gap electron mobility",
    "orbita eliosincrona satellite osservazione",
    "spectroscopie infrarouge exoplanète atmosphère",
    "quantum entanglement bell inequality violation",
]


def best_similarity(retriever: HybridRetriever, embedder: Embedder, q: str,
                    domain: str = "pilot") -> float:
    vec = retriever.vector_search(embedder.encode([q])[0], top_k=25, domain=domain)
    return max((s for _, s in vec), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerable", type=Path, default=Path("../eval/neutral-qa-v1.json"))
    ap.add_argument("--golden", type=Path, default=Path("../eval/golden-qa-v1.json"),
                    help="its 'unanswerable' pairs are added to the negative set")
    ap.add_argument("--json", type=Path,
                    default=Path("../evidence/T71/refusal-calibration.json"))
    args = ap.parse_args()

    pos = [p["query"] for p in json.loads(args.answerable.read_text())["pairs"]]
    neg = list(OUT_OF_CORPUS)
    if args.golden.exists():
        neg += [p["query"] for p in json.loads(args.golden.read_text())["pairs"]
                if p.get("kind") == "unanswerable"]

    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    try:
        pos_s = [best_similarity(retriever, embedder, q) for q in pos]
        neg_s = [best_similarity(retriever, embedder, q) for q in neg]
    finally:
        loader.close()

    def band(v):
        return {"n": len(v), "min": round(min(v), 4), "p5": round(sorted(v)[len(v)//20], 4),
                "median": round(statistics.median(v), 4), "max": round(max(v), 4)}

    print(f"nel corpus      {band(pos_s)}")
    print(f"fuori corpus    {band(neg_s)}")
    print(f"\n{'soglia':>7}{'rifiutate fuori corpus':>26}{'perse nel corpus':>20}")

    rows = []
    for i in range(60, 100):
        floor = i / 100
        refused = sum(1 for s in neg_s if s < floor)
        lost = sum(1 for s in pos_s if s < floor)
        rows.append({"floor": floor, "refused": refused, "of_neg": len(neg_s),
                     "lost": lost, "of_pos": len(pos_s),
                     "refused_pct": round(100 * refused / len(neg_s), 1),
                     "lost_pct": round(100 * lost / len(pos_s), 1)})

    full = [r for r in rows if r["refused"] == r["of_neg"]]
    chosen = min(full, key=lambda r: r["lost"]) if full else max(rows, key=lambda r: r["refused"])
    for r in rows:
        if r["floor"] * 100 % 2 and r is not chosen:
            continue
        mark = "  <- scelta" if r is chosen else ""
        print(f"{r['floor']:>7.2f}{r['refused']:>18}/{r['of_neg']:<7}"
              f"{r['lost']:>13}/{r['of_pos']:<6}{mark}")

    verdict = {
        "in_corpus": band(pos_s), "out_of_corpus": band(neg_s),
        "sweep": rows, "chosen_floor": chosen["floor"],
        "chosen_refused": f"{chosen['refused']}/{chosen['of_neg']}",
        "chosen_recall_cost": f"{chosen['lost']}/{chosen['of_pos']}",
        "separable": bool(full),
        "note": "no floor separates the two bands perfectly when 'separable' is "
                "false; the choice is then a policy decision, and Q1 outranks recall.",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"\nsoglia proposta {chosen['floor']:.2f} — rifiuta {verdict['chosen_refused']} "
          f"fuori corpus, costa {verdict['chosen_recall_cost']} nel corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
