#!/usr/bin/env python3
"""T74 — adjudicate the three MSC recipe cards against one loaded graph.

One arm per run: point NEO4J_URI at the instance to test (7688 ENTERPRIPHY,
7689 upstream) and pass --arm so the output says which architecture answered.
Accuracy is scored against eval/cards/expected-T74.json, whose verdicts come
from reading the source PDFs — never from either extraction. An expected
NOT_SUPPORTED is satisfied by CONTRADICTED or NOT_FOUND: both mean "the graph
did not confirm a claim the source does not make".
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from graphify_ent.embed import Embedder
from graphify_ent.loader import Neo4jLoader
from graphify_ent.retrieval import HybridRetriever
from graphify_ent.verify import SUPPORTED, Claim, Verifier

CARDS = ["meatballs-6884", "joconde-8950", "strawberry-jelly-15915"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["enterpriphy", "upstream"])
    ap.add_argument("--cards-dir", type=Path, default=Path("../eval/cards"))
    ap.add_argument("--expected", type=Path,
                    default=Path("../eval/cards/expected-T74.json"))
    ap.add_argument("--domain", default="pilot")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"../evidence/T74/cards-{args.arm}.json")

    exp = {}
    for e in json.loads(args.expected.read_text())["expected"]:
        exp[(e["card"], e["subject"], e["aspect"])] = e

    loader = Neo4jLoader()
    print(f"arm={args.arm}  graph={loader.uri}")
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    embedder.encode(["warm up"])
    verifier = Verifier(retriever, embed_fn=lambda q: embedder.encode([q])[0],
                        domain=args.domain)

    rows = []
    t_all = time.perf_counter()
    try:
        for name in CARDS:
            card = json.loads((args.cards_dir / f"{name}.json").read_text())
            claims = [Claim(**c) for c in card["claims"]]
            for f in verifier.check_all(claims):
                e = exp.get((name, f.claim.subject, f.claim.aspect))
                expected = e["expected"] if e else "?"
                correct = (f.verdict == SUPPORTED) == (expected == "SUPPORTED")
                rows.append({"card": name, **f.as_dict(),
                             "expected": expected, "correct": correct,
                             "ground_truth": e["ground_truth"] if e else ""})
                mark = "OK " if correct else "MISS"
                print(f"  {mark} [{f.verdict:<12}] atteso {expected:<13} "
                      f"{f.claim.subject[:24]:<26} {f.claim.aspect[:34]:<36} "
                      f"{f.latency_ms:6.0f} ms", flush=True)
    finally:
        loader.close()
    wall_ms = (time.perf_counter() - t_all) * 1000

    lat = [r["latency_ms"] for r in rows]
    n_ok = sum(r["correct"] for r in rows)
    sup_exp = [r for r in rows if r["expected"] == "SUPPORTED"]
    uns_exp = [r for r in rows if r["expected"] == "NOT_SUPPORTED"]
    report = {
        "arm": args.arm,
        "claims": len(rows),
        "correct": n_ok,
        "accuracy_pct": round(100 * n_ok / max(len(rows), 1), 1),
        "supported_recall": f"{sum(r['correct'] for r in sup_exp)}/{len(sup_exp)}",
        "unsupported_rejection": f"{sum(r['correct'] for r in uns_exp)}/{len(uns_exp)}",
        "false_confirmations": sum(1 for r in uns_exp if not r["correct"]),
        "latency_ms_mean": round(statistics.mean(lat), 1) if lat else None,
        "latency_ms_p95": round(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)], 1)
        if lat else None,
        "total_ms": round(wall_ms, 1),
        "verdicts": {v: sum(1 for r in rows if r["verdict"] == v)
                     for v in ("SUPPORTED", "CONTRADICTED", "NOT_FOUND")},
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{args.arm}: {n_ok}/{len(rows)} corrette "
          f"({report['accuracy_pct']}%), conferme false {report['false_confirmations']}, "
          f"latenza media {report['latency_ms_mean']} ms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
