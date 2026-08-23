#!/usr/bin/env python3
"""Phase 3.2 / 6.7 — evaluation harness: Q1 faithfulness, Q2, Q3 recall@10.

Ablations (execution plan §3.2): fulltext-only, vector-only, hybrid,
hybrid+graph, hybrid+graph+expansion — each with p95 latency.

Metric definitions, and exactly what each one does and does not prove:

Q1 faithfulness — for every hit the server serialized, re-check that its
   `evidence` is a literal substring of the node's stored source text, and that
   an unanswerable query produced the explicit refusal instead of hits. This is
   a machine check on real data: it measures the property the architecture
   guarantees (evidence binding + refusal path), not a model's self-report.

Q2 correctness — the plan defines this over generated answers. No answer
   generator runs here (no LLM credentials), so this harness reports a
   **proxy**: does the retrieved, token-budgeted context actually contain the
   ground-truth answer span? A generator cannot be correct if the span never
   reached its context, so this is an upper bound on Q2, reported as
   `q2_context_contains_answer` and never as Q2 itself.

Q3 recall@10 — does the ground-truth document appear in the top-10 fused
   results? Measured separately for monolingual and cross-language queries.

Usage:
  python tools/run_eval.py --golden ../eval/golden-qa-v1.json --json OUT.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from graphify_ent.embed import Embedder
from graphify_ent.loader import Neo4jLoader
from graphify_ent.retrieval import (
    DEFAULT_TOKEN_BUDGET,
    HybridRetriever,
    serialize_context,
    verify_evidence_binding,
)

ABLATIONS = {
    "fulltext_only": {"channels": ("fulltext",), "hops": 0, "use_embedding": False},
    "vector_only": {"channels": ("vector",), "hops": 0, "use_embedding": True},
    "hybrid": {"channels": ("vector", "fulltext"), "hops": 0, "use_embedding": True},
    "hybrid_graph": {"channels": ("vector", "fulltext", "graph"), "hops": 1,
                     "use_embedding": True},
    "hybrid_graph_expansion": {"channels": ("vector", "fulltext", "graph"), "hops": 1,
                               "use_embedding": True, "expansion": True},
}

# Built from the cross-language term table: the deterministic tier-(a) glossary
# that Phase 6.0 will later generate from the :Concept layer.
GLOSSARY = {
    "eggplant": ["aubergine", "melanzana"], "aubergine": ["eggplant", "melanzana"],
    "melanzana": ["aubergine", "eggplant"],
    "butter": ["beurre", "burro"], "beurre": ["butter", "burro"], "burro": ["butter", "beurre"],
    "flour": ["farine", "farina"], "farine": ["flour", "farina"], "farina": ["flour", "farine"],
    "egg": ["oeuf", "uovo"], "oeuf": ["egg", "uovo"], "uovo": ["egg", "oeuf"],
    "sugar": ["sucre", "zucchero"], "sucre": ["sugar", "zucchero"],
    "zucchero": ["sugar", "sucre"],
    "salt": ["sel", "sale"], "sel": ["salt", "sale"], "sale": ["salt", "sel"],
    "cream": ["crème", "panna"], "crème": ["cream", "panna"], "panna": ["cream", "crème"],
    "chicken": ["poulet", "pollo"], "poulet": ["chicken", "pollo"],
    "pollo": ["chicken", "poulet"],
    "onion": ["oignon", "cipolla"], "oignon": ["onion", "cipolla"],
    "cipolla": ["onion", "oignon"],
    "mushroom": ["champignon", "fungo"], "champignon": ["mushroom", "fungo"],
    "fungo": ["mushroom", "champignon"],
    "sauce": ["salsa"], "salsa": ["sauce"],
    "fish": ["poisson", "pesce"], "poisson": ["fish", "pesce"], "pesce": ["fish", "poisson"],
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def run_ablation(retriever, embedder, pairs, name, cfg, top_n=10):
    """Run one configuration over the whole golden set."""
    latencies, results = [], []
    answerable = [p for p in pairs if p["kind"] != "unanswerable"]
    unanswerable = [p for p in pairs if p["kind"] == "unanswerable"]

    glossary_backup = retriever.glossary
    retriever.glossary = GLOSSARY if cfg.get("expansion") else {}

    try:
        for p in pairs:
            emb = embedder.encode([p["query"]])[0] if cfg["use_embedding"] else None
            t0 = time.perf_counter()
            res = retriever.query(
                p["query"],
                embedding=emb,
                channels=cfg["channels"],
                hops=cfg["hops"],
                domain="pilot",
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            top_hits = res.hits[:top_n]
            top_docs = []
            for h in top_hits:
                if h.source_file not in top_docs:
                    top_docs.append(h.source_file)

            context = serialize_context(top_hits, token_budget=DEFAULT_TOKEN_BUDGET)
            span = norm(p.get("answer_span") or "")
            results.append({
                "id": p["id"],
                "kind": p["kind"],
                "refused": res.refused,
                "doc_hit": bool(p.get("ground_truth_doc")) and p["ground_truth_doc"] in top_docs,
                "node_hit": bool(p.get("ground_truth_node"))
                and any(h.node_id == p["ground_truth_node"] for h in top_hits),
                "context_has_answer": bool(span) and span[:120] in norm(context),
                "n_hits": len(res.hits),
                "hits": [(h.node_id, h.source_file, h.evidence) for h in top_hits],
            })
    finally:
        retriever.glossary = glossary_backup

    def recall(kind: str | None = None) -> float | None:
        sel = [r for r in results
               if r["kind"] != "unanswerable" and (kind is None or r["kind"] == kind)]
        return round(100 * sum(r["doc_hit"] for r in sel) / len(sel), 2) if sel else None

    ctx = [r for r in results if r["kind"] != "unanswerable"]
    refusals_correct = sum(
        1 for r in results if r["kind"] == "unanswerable" and (r["refused"] or r["n_hits"] == 0)
    )

    return {
        "ablation": name,
        "queries": len(pairs),
        "q3_recall_at_10_all": recall(),
        "q3_recall_at_10_monolingual": recall("monolingual"),
        "q3_recall_at_10_cross_language": recall("cross_language"),
        "q3_recall_at_10_acronym": recall("acronym"),
        "node_level_recall_at_10": round(
            100 * sum(r["node_hit"] for r in ctx) / len(ctx), 2) if ctx else None,
        "q2_context_contains_answer_pct": round(
            100 * sum(r["context_has_answer"] for r in ctx) / len(ctx), 2) if ctx else None,
        "unanswerable_refused": f"{refusals_correct}/{len(unanswerable)}",
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else None,
        "latency_ms_p95": round(
            statistics.quantiles(latencies, n=20)[18], 1) if len(latencies) > 1 else None,
        "_per_query": results,
    }


def measure_q1(retriever, results_for_best: list[dict]) -> dict:
    """Q1: every serialized claim must be a literal substring of its source."""
    node_ids = [nid for r in results_for_best for (nid, _, _) in r["hits"]]
    props = retriever.hydrate(list(dict.fromkeys(node_ids)))

    checked = unsupported = 0
    offenders = []
    for r in results_for_best:
        for nid, _src, evidence in r["hits"]:
            p = props.get(nid)
            if not p:
                unsupported += 1
                checked += 1
                offenders.append({"node": nid, "reason": "hit not resolvable to a source node"})
                continue
            source_text = p.get("text_excerpt") or ""
            checked += 1
            if not verify_evidence_binding(evidence, source_text):
                unsupported += 1
                offenders.append({"node": nid, "reason": "evidence not a substring of source"})

    unanswered = [r for r in results_for_best if r["kind"] == "unanswerable"]
    bad_refusals = [r for r in unanswered if not (r["refused"] or r["n_hits"] == 0)]

    return {
        "claims_checked": checked,
        "unsupported_claims": unsupported,
        "faithfulness_pct": round(100 * (checked - unsupported) / checked, 3) if checked else None,
        "unanswerable_incorrectly_answered": len(bad_refusals),
        "offenders_sample": offenders[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--only", default=None, help="run a single ablation by name")
    args = ap.parse_args()

    golden = json.loads(args.golden.read_text())
    pairs = golden["pairs"]

    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()

    try:
        report = {"golden_version": golden.get("version"),
                  "corpus": golden.get("corpus"),
                  "provenance": golden.get("provenance"),
                  "counts": golden.get("counts"),
                  "ablations": []}

        for name, cfg in ABLATIONS.items():
            if args.only and name != args.only:
                continue
            print(f"\n--- {name} ---", flush=True)
            res = run_ablation(retriever, embedder, pairs, name, cfg)
            per_query = res.pop("_per_query")
            print(json.dumps(res, indent=2, ensure_ascii=False))
            if name == "hybrid_graph_expansion":
                report["q1"] = measure_q1(retriever, per_query)
                print("Q1:", json.dumps(report["q1"], indent=2))
            report["ablations"].append(res)

        best = next((a for a in report["ablations"]
                     if a["ablation"] == "hybrid_graph_expansion"), None)
        ft = next((a for a in report["ablations"] if a["ablation"] == "fulltext_only"), None)
        if best and ft:
            report["acceptance"] = {
                "Q1_faithfulness_target_pct": 99.5,
                "Q1_measured_pct": report.get("q1", {}).get("faithfulness_pct"),
                "Q1_pass": (report.get("q1", {}).get("faithfulness_pct") or 0) >= 99.5,
                "Q3_recall_mono_target_pct": 95,
                "Q3_recall_mono_measured": best["q3_recall_at_10_monolingual"],
                "Q3_recall_cross_target_pct": 90,
                "Q3_recall_cross_measured": best["q3_recall_at_10_cross_language"],
                "p95_latency_target_ms": 800,
                "p95_latency_measured_ms": best["latency_ms_p95"],
                "p95_pass": (best["latency_ms_p95"] or 1e9) < 800,
                "hybrid_beats_fulltext_cross_language": (
                    (best["q3_recall_at_10_cross_language"] or 0)
                    > (ft["q3_recall_at_10_cross_language"] or 0)
                ),
            }
            print("\n=== ACCEPTANCE ===")
            print(json.dumps(report["acceptance"], indent=2))

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    finally:
        loader.close()


if __name__ == "__main__":
    raise SystemExit(main())
