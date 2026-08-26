#!/usr/bin/env python3
"""Re-measure the refusal floor for the encoder currently configured.

The floor is a property of the corpus AND the encoder together. Carrying a
number calibrated for a different model is how a system stops refusing without
anyone noticing — measured once already (T73, defect 8), so a model swap must
never ship without re-running this.

Method: take the in-corpus queries from the Q2 golden set and a fixed set of
out-of-corpus questions in three languages, record each one's best vector
similarity, then report the two bands and what each candidate floor would cost.
The chosen floor should sit in the gap between the bands, not on either edge.

    python tools/calibrate_refusal.py            # uses EMBED_MODEL from env
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

#: Deliberately far from a culinary corpus, and deliberately multilingual: a
#: floor that only rejects English nonsense is not a floor. The last two are
#: lexical traps — "stock" is a cooking fund, "service" is on every page.
OUT_OF_CORPUS = [
    "kubernetes ingress controller tls termination",
    "employee stock option vesting schedule for kitchen staff",
    "what is the mortgage interest deduction limit",
    "how do I reset my active directory password",
    "quelle est la capitale administrative de la Bolivie",
    "quel est le taux de cotisation retraite en France en 2026",
    "comment installer un certificat SSL sur nginx",
    "qual è la posologia del paracetamolo per adulti",
    "come si calcola l'IMU sulla seconda casa",
    "quali sono le regole del fuorigioco nel calcio",
    "python asyncio event loop deadlock debugging",
    "difference between mitosis and meiosis",
    "che cosa prevede il GDPR per il consenso dei minori",
    "quel est le classement de la Ligue 1 cette saison",
    "how to change the timing belt on a diesel engine",
    "storia della guerra dei trent'anni cause e conseguenze",
    "best practices for terraform state locking",
    "quelle est la composition chimique du polystyrène",
    "customer service ticket escalation matrix",
    "annual depreciation schedule for restaurant equipment",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("../eval/q2-golden-v1.json"))
    ap.add_argument("--domain", default="pilot")
    ap.add_argument("--json", type=Path,
                    default=Path("../evidence/T83/refusal-calibration.json"))
    args = ap.parse_args()

    from graphify_ent.embed import Embedder
    from graphify_ent.loader import Neo4jLoader
    from graphify_ent.retrieval import HybridRetriever

    pairs = json.loads(args.golden.read_text())["pairs"]
    inside = [p["query"] for p in pairs if p["kind"] != "unanswerable"]
    outside = OUT_OF_CORPUS + [p["query"] for p in pairs if p["kind"] == "unanswerable"]

    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    embedder.encode(["warm up"])

    def best_similarity(q: str) -> float:
        hits = retriever.vector_search(embedder.encode([q])[0], top_k=5,
                                       domain=args.domain)
        return max((s for _, s in hits), default=0.0)

    try:
        ins = sorted(best_similarity(q) for q in inside)
        outs = sorted(best_similarity(q) for q in outside)
    finally:
        loader.close()

    def band(v):
        return {"n": len(v), "min": round(v[0], 3),
                "p5": round(v[max(0, int(0.05 * len(v)) - 1)], 3),
                "median": round(statistics.median(v), 3), "max": round(v[-1], 3)}

    table = []
    for floor in [round(x / 100, 2) for x in range(40, 96, 5)]:
        table.append({"floor": floor,
                      "out_of_corpus_refused": f"{sum(1 for s in outs if s < floor)}/{len(outs)}",
                      "in_corpus_lost": f"{sum(1 for s in ins if s < floor)}/{len(ins)}"})
    clean = [r for r in table
             if r["out_of_corpus_refused"].split("/")[0] == str(len(outs))
             and r["in_corpus_lost"].startswith("0/")]
    report = {
        "encoder": os.environ.get("EMBED_MODEL", "(default)"),
        "dim": os.environ.get("EMBED_DIM", "(default)"),
        "in_corpus": band(ins), "out_of_corpus": band(outs),
        "separated": ins[0] > outs[-1],
        "gap": round(ins[0] - outs[-1], 3),
        "sweep": table,
        "floors_with_full_coverage_and_no_loss": [r["floor"] for r in clean],
        "recommended_floor": (round((ins[0] + outs[-1]) / 2, 2)
                              if ins[0] > outs[-1] else None),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
