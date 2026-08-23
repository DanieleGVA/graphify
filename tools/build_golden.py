#!/usr/bin/env python3
"""Build the golden Q&A set from real pilot passages.

**Provenance note (matters for how the numbers should be read).** The execution
plan calls for 30–50 pairs authored *once, with Daniele*, because domain ground
truth needs a human. No human is available on this run, so this tool derives the
set mechanically from passages that actually exist in the six PDFs: each query
is built from a distinctive term found in a specific passage, and the ground
truth is that passage's source document and node id.

That makes recall@10 (Q3) an honest measurement — the target document provably
contains the answer — while keeping the set reproducible and versioned. It does
NOT make it a human-validated set: the queries are keyword-shaped, not the
natural-language questions a curator would write. Treat Q3 here as a floor, and
re-run against a human-authored set before quoting a G3 number.

Cross-language pairs are constructed explicitly: a query drawn from a French
document's content is also emitted in English and Italian translation of its
key term, so the cross-lingual leg exercises the semantic channel rather than
lexical overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# Culinary terms with reliable IT/EN/FR translations — used to build the
# cross-language leg. Deliberately small and hand-checked.
CROSS_LANG_TERMS = [
    {"en": "eggplant", "it": "melanzana", "fr": "aubergine"},
    {"en": "butter", "it": "burro", "fr": "beurre"},
    {"en": "flour", "it": "farina", "fr": "farine"},
    {"en": "egg", "it": "uovo", "fr": "oeuf"},
    {"en": "sugar", "it": "zucchero", "fr": "sucre"},
    {"en": "salt", "it": "sale", "fr": "sel"},
    {"en": "cream", "it": "panna", "fr": "crème"},
    {"en": "chicken", "it": "pollo", "fr": "poulet"},
    {"en": "onion", "it": "cipolla", "fr": "oignon"},
    {"en": "mushroom", "it": "fungo", "fr": "champignon"},
    {"en": "sauce", "it": "salsa", "fr": "sauce"},
    {"en": "fish", "it": "pesce", "fr": "poisson"},
]

# Abbreviation/acronym-heavy probes (≥5 required by the plan).
ACRONYM_QUERIES = [
    "tbsp tsp measurement conversion",
    "oz lb weight conversion table",
    "ml cl dl liquid measure",
    "F C oven temperature conversion",
    "AP flour all purpose",
]

_STOP = set("""the and for with from that this into over under your their there
which when what while about above below very more most such than then them they
les des une aux par pour dans sur avec est sont plus tout tous cette ces
gli che con per una del della delle dei alla alle sono come anche""".split())


def distinctive_terms(text: str, min_len: int = 6, top: int = 3) -> list[str]:
    """Rare-ish alphabetic tokens that make a passage findable."""
    words = re.findall(r"[A-Za-zÀ-ÿ]{%d,20}" % min_len, text)
    seen, out = set(), []
    for w in words:
        lw = w.lower()
        if lw in _STOP or lw in seen:
            continue
        seen.add(lw)
        out.append(lw)
        if len(out) >= top:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", type=Path, help="pilot-graph.json from pilot_extract")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-doc", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    graph = json.loads(args.graph.read_text())
    sections = [
        n for n in graph["nodes"]
        if n.get("source_location") and len(n.get("text_excerpt") or "") > 400
    ]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for n in sections:
        by_doc[n["source_file"]].append(n)

    rng = random.Random(args.seed)
    pairs: list[dict] = []

    # --- monolingual: distinctive-term probes anchored to one passage -------
    for doc, nodes in sorted(by_doc.items()):
        picks = rng.sample(nodes, min(args.per_doc, len(nodes)))
        for node in picks:
            terms = distinctive_terms(node["text_excerpt"])
            if len(terms) < 2:
                continue
            query = " ".join(terms[:3])
            pairs.append({
                "id": "q_" + hashlib.sha1(f"{doc}{query}".encode()).hexdigest()[:10],
                "query": query,
                "query_lang": node.get("lang") or "en",
                "kind": "monolingual",
                "ground_truth_doc": doc,
                "ground_truth_node": node["id"],
                "answer_span": node["evidence"][:200],
                "source_location": node.get("source_location"),
            })

    # --- cross-language: same concept, query in a different language -------
    lang_docs: dict[str, list[dict]] = defaultdict(list)
    for n in sections:
        lang_docs[n.get("lang") or "en"].append(n)

    for term in CROSS_LANG_TERMS:
        for target_lang, query_langs in (("fr", ("en", "it")), ("en", ("it", "fr"))):
            candidates = [
                n for n in lang_docs.get(target_lang, [])
                if term[target_lang].lower() in (n.get("text_excerpt") or "").lower()
            ]
            if not candidates:
                continue
            node = rng.choice(candidates)
            for ql in query_langs:
                pairs.append({
                    "id": "qx_" + hashlib.sha1(
                        f"{node['id']}{term[ql]}{ql}".encode()).hexdigest()[:10],
                    "query": term[ql],
                    "query_lang": ql,
                    "kind": "cross_language",
                    "target_lang": target_lang,
                    "concept": term["en"],
                    "ground_truth_doc": node["source_file"],
                    "ground_truth_node": node["id"],
                    "answer_span": node["evidence"][:200],
                    "source_location": node.get("source_location"),
                })

    # --- acronym-heavy probes (ground truth: any doc containing the terms) --
    for q in ACRONYM_QUERIES:
        toks = [t for t in q.split() if len(t) <= 4]
        hit = next(
            (n for n in sections
             if all(re.search(rf"\b{re.escape(t)}\b", n.get("text_excerpt") or "",
                              re.IGNORECASE) for t in toks[:2])),
            None,
        )
        if hit:
            pairs.append({
                "id": "qa_" + hashlib.sha1(q.encode()).hexdigest()[:10],
                "query": q,
                "query_lang": "en",
                "kind": "acronym",
                "ground_truth_doc": hit["source_file"],
                "ground_truth_node": hit["id"],
                "answer_span": hit["evidence"][:200],
                "source_location": hit.get("source_location"),
            })

    # --- unanswerable probes: the refusal path must fire -------------------
    for q in [
        "quarterly revenue forecast for the Zurich office",
        "kubernetes ingress controller misconfiguration",
        "employee stock option vesting schedule",
    ]:
        pairs.append({
            "id": "qn_" + hashlib.sha1(q.encode()).hexdigest()[:10],
            "query": q,
            "query_lang": "en",
            "kind": "unanswerable",
            "ground_truth_doc": None,
            "ground_truth_node": None,
            "answer_span": None,
        })

    payload = {
        "version": 1,
        "corpus": "foodmdm-pilot-pdf-only",
        "provenance": "auto-derived from real passages; NOT human-authored (see tool docstring)",
        "counts": {
            "total": len(pairs),
            "monolingual": sum(1 for p in pairs if p["kind"] == "monolingual"),
            "cross_language": sum(1 for p in pairs if p["kind"] == "cross_language"),
            "acronym": sum(1 for p in pairs if p["kind"] == "acronym"),
            "unanswerable": sum(1 for p in pairs if p["kind"] == "unanswerable"),
        },
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload["counts"], indent=2))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
