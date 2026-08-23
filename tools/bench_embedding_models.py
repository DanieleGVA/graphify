#!/usr/bin/env python3
"""Pick the embedding model on the three axes that matter: accuracy, cost, speed.

Measured on the real pilot graph and the real golden set, through the real
hybrid pipeline — the lexical channel is model-independent, so it is fetched
once from Neo4j and RRF-fused with each model's vector ranking. Comparing
vector-only would flatter models the pipeline never uses alone.

Per model:

  accuracy  recall@10 (all / monolingual / cross-language) after RRF fusion,
            plus **refusal separability** — can one floor reject every
            out-of-corpus query without losing answerable ones? MTEB does not
            measure this, and Q1 depends on it entirely.
  cost      local inference is $0 marginal; the real recurring costs are vector
            storage (dimension x nodes) and re-embedding time.
  speed     nodes/s on this machine, and query encode latency.

Usage: python tools/bench_embedding_models.py --json out.json [--models a,b,c]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RRF_K = 60

# prefixes: e5 and Qwen3 are instruction-tuned and degrade badly without them.
MODELS = {
    "BAAI/bge-m3": {"q": "", "p": "", "note": "current baseline"},
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {"q": "", "p": ""},
    "intfloat/multilingual-e5-small": {"q": "query: ", "p": "passage: "},
    "intfloat/multilingual-e5-base": {"q": "query: ", "p": "passage: "},
    "intfloat/multilingual-e5-large-instruct": {"q": "query: ", "p": "passage: "},
    "Qwen/Qwen3-Embedding-0.6B": {"q": "", "p": "", "prompt_name": "query"},
    "Alibaba-NLP/gte-multilingual-base": {"q": "", "p": "", "trust": True},
    "jinaai/jina-embeddings-v3": {"q": "", "p": "", "trust": True},
    "sentence-transformers/distiluse-base-multilingual-cased-v2": {"q": "", "p": ""},
}

GLOSSARY = {
    "eggplant": ["aubergine", "melanzana"], "aubergine": ["eggplant", "melanzana"],
    "melanzana": ["aubergine", "eggplant"], "butter": ["beurre", "burro"],
    "beurre": ["butter", "burro"], "burro": ["butter", "beurre"],
    "flour": ["farine", "farina"], "farine": ["flour", "farina"],
    "farina": ["flour", "farine"], "egg": ["oeuf", "uovo"], "oeuf": ["egg", "uovo"],
    "uovo": ["egg", "oeuf"], "sugar": ["sucre", "zucchero"],
    "sucre": ["sugar", "zucchero"], "zucchero": ["sugar", "sucre"],
    "salt": ["sel", "sale"], "sel": ["salt", "sale"], "sale": ["salt", "sel"],
    "cream": ["crème", "panna"], "crème": ["cream", "panna"],
    "panna": ["cream", "crème"], "chicken": ["poulet", "pollo"],
    "poulet": ["chicken", "pollo"], "pollo": ["chicken", "poulet"],
    "onion": ["oignon", "cipolla"], "oignon": ["onion", "cipolla"],
    "cipolla": ["onion", "oignon"], "mushroom": ["champignon", "fungo"],
    "champignon": ["mushroom", "fungo"], "fungo": ["mushroom", "champignon"],
    "sauce": ["salsa"], "salsa": ["sauce"], "fish": ["poisson", "pesce"],
    "poisson": ["fish", "pesce"], "pesce": ["fish", "poisson"],
}


def expand(query: str) -> list[str]:
    import re
    out = [query]
    low = query.lower()
    for term, alts in GLOSSARY.items():
        if term in low:
            for alt in alts:
                v = re.sub(re.escape(term), alt, query, flags=re.IGNORECASE)
                if v not in out:
                    out.append(v)
    return out


def fetch_lexical(pairs, node_ids) -> dict[str, list[str]]:
    """Fulltext rankings — identical for every model, so fetched once."""
    from graphify_ent.loader import Neo4jLoader
    from graphify_ent.retrieval import HybridRetriever

    loader = Neo4jLoader()
    r = HybridRetriever(loader)
    known = set(node_ids)
    out = {}
    for p in pairs:
        ranked: list[str] = []
        for variant in expand(p["query"]):
            for nid, _ in r.fulltext_search(variant, top_k=25, domain="pilot"):
                if nid in known and nid not in ranked:
                    ranked.append(nid)
        out[p["id"]] = ranked
    loader.close()
    return out


def rrf(vector_ids: list[str], lexical_ids: list[str], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for lst in (vector_ids, lexical_ids):
        for rank, nid in enumerate(lst, start=1):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=Path("/tmp/pilot-graph.json"))
    ap.add_argument("--golden", type=Path, default=Path("../eval/golden-qa-v1.json"))
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--models", default=None)
    args = ap.parse_args()

    graph = json.loads(args.graph.read_text())
    nodes = [n for n in graph["nodes"] if n.get("text_excerpt")]
    node_ids = [n["id"] for n in nodes]
    node_docs = [n["source_file"] for n in nodes]
    node_texts = [(n.get("label", "") + "\n" + n["text_excerpt"])[:1000] for n in nodes]

    golden = json.loads(args.golden.read_text())
    pairs = golden["pairs"]
    answerable = [p for p in pairs if p["kind"] != "unanswerable"]

    print(f"corpus {len(nodes)} nodi · golden {len(pairs)} query "
          f"({len(answerable)} con risposta, {len(pairs)-len(answerable)} senza)\n")

    print("recupero il canale lessicale una volta sola (indipendente dal modello)...")
    lexical = fetch_lexical(pairs, node_ids)
    print("fatto.\n")

    selected = ([m.strip() for m in args.models.split(",")] if args.models
                else list(MODELS))
    results = {}

    for name in selected:
        cfg = MODELS.get(name, {"q": "", "p": ""})
        print(f"--- {name} ---", flush=True)
        try:
            from sentence_transformers import SentenceTransformer
            kw = {"trust_remote_code": True} if cfg.get("trust") else {}
            t0 = time.perf_counter()
            model = SentenceTransformer(name, device="mps", **kw)
            load_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            P = model.encode([cfg["p"] + t for t in node_texts], batch_size=64,
                             normalize_embeddings=True, show_progress_bar=False,
                             convert_to_numpy=True)
            embed_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            Q = model.encode([cfg["q"] + p["query"] for p in pairs], batch_size=64,
                             normalize_embeddings=True, show_progress_bar=False,
                             convert_to_numpy=True)
            query_s = (time.perf_counter() - t0) / len(pairs)
            dim = int(P.shape[1])
            params = sum(p.numel() for p in model.parameters()) / 1e6
            del model
        except Exception as exc:
            print(f"  FALLITO: {type(exc).__name__}: {str(exc)[:160]}\n")
            results[name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            continue

        S = Q @ P.T
        hits = {"all": 0, "monolingual": 0, "cross_language": 0, "acronym": 0}
        totals = {"all": 0, "monolingual": 0, "cross_language": 0, "acronym": 0}
        vec_max_ans, vec_max_una = [], []

        for i, p in enumerate(pairs):
            row = S[i]
            top_vec = [node_ids[j] for j in np.argsort(-row)[:25]]
            if p["kind"] == "unanswerable":
                vec_max_una.append(float(row.max()))
                continue
            vec_max_ans.append(float(row.max()))
            fused = rrf(top_vec, lexical.get(p["id"], []))
            top = sorted(fused, key=lambda n: -fused[n])[:10]
            docs, idx = [], {n: k for k, n in enumerate(node_ids)}
            for nid in top:
                d = node_docs[idx[nid]]
                if d not in docs:
                    docs.append(d)
            ok = p["ground_truth_doc"] in docs
            hits["all"] += ok
            totals["all"] += 1
            if p["kind"] in hits:
                hits[p["kind"]] += ok
                totals[p["kind"]] += 1

        # Refusal separability: the floor that rejects every unanswerable query,
        # and what it costs in answerable ones.
        floor = max(vec_max_una) + 1e-6 if vec_max_una else 0.0
        lost = sum(1 for x in vec_max_ans if x < floor)
        separable = min(vec_max_ans) > max(vec_max_una) if vec_max_ans and vec_max_una else False

        rec = {k: round(100 * hits[k] / totals[k], 1) if totals[k] else None for k in hits}
        results[name] = {
            "recall_all": rec["all"], "recall_mono": rec["monolingual"],
            "recall_cross": rec["cross_language"], "recall_acronym": rec["acronym"],
            "refusal_floor": round(floor, 4),
            "answerable_lost_at_full_refusal": f"{lost}/{len(vec_max_ans)}",
            "answerable_lost_pct": round(100 * lost / len(vec_max_ans), 1) if vec_max_ans else None,
            "cleanly_separable": separable,
            "dim": dim, "params_m": round(params, 0),
            "embed_s_3187": round(embed_s, 1),
            "nodes_per_s": round(len(nodes) / embed_s, 1),
            "query_ms": round(query_s * 1000, 1),
            "load_s": round(load_s, 1),
            "storage_mb_per_1M_nodes": round(dim * 4 / 1e6 * 1e6 / 1e6, 1),
            "hours_to_embed_1M": round(1e6 / (len(nodes) / embed_s) / 3600, 2),
        }
        r = results[name]
        print(f"  recall all={r['recall_all']} mono={r['recall_mono']} "
              f"cross={r['recall_cross']} acr={r['recall_acronym']}")
        print(f"  rifiuto: perde {r['answerable_lost_at_full_refusal']} "
              f"({r['answerable_lost_pct']}%), separazione netta={r['cleanly_separable']}")
        print(f"  {r['nodes_per_s']} nodi/s · dim {r['dim']} · {r['params_m']}M par · "
              f"{r['hours_to_embed_1M']}h per 1M nodi\n", flush=True)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"salvato -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
