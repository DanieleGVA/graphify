#!/usr/bin/env python3
"""Validate a recipe card against the corpus when nothing declares its source.

The merged card export carries no `Canon` field: the reference has to be found,
not checked. Two ways of finding it are implemented here and both are reported,
because one of them is measured to fail and the failure is the point.

**fingerprint-first** — rank every recipe page of the corpus by the three
criteria (rarity-weighted ingredient proportions, technique set and sequence,
title). This is the criterion established in T96 and it does not survive
contact with real cards: measured on the Pareto export, whose cards declare
their reference work, the dishes whose book the corpus does NOT hold scored
*higher* than the true matches (median 0.515 against 0.393; rarity 0.685
against 0.466), and no derived signal — margin over the runner-up, rarity mass
explained, the query's own discriminating power — separates the two. A card is
a purchasing spec made of sub-recipes and yields; a cookbook page teaches a
dish from raw ingredients. They agree on the common things and differ on
everything that identifies them.

**retrieval-first** — ask the graph for the pages that TALK about this dish
(the evidence lane, measured at 95.5% grounding on the canon benchmark), then
use the fingerprint only to rank and explain those few. Finding is a text
problem; the fingerprint is a verification instrument, and using it as a search
key was the inversion this tool exists to measure.

Every answer carries its evidence, and a card whose reference cannot be
supported gets an explicit refusal rather than the nearest page.

    python tools/validate_cards.py --cards ../tests/Recipe_Cards_Merged_v001_abstract.pdf
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
from graphify_ent.recipes.cards import load_cards
from graphify_ent.recipes.ingredients import Registry, proportions
from graphify_ent.recipes.match import (
    CorpusIndex,
    RecipeQuery,
    confidence,
    explain,
)
from graphify_ent.recipes.techniques import techniques_in
from graphify_ent.retrieval import HybridRetriever

#: Words that make a card title a shelf label rather than a dish name.
_TITLE_NOISE = re.compile(r"\([^)]*\)|\b(YC|CC|MD|VG|NSA|AA|CREW|FP|LUNCH|BUFFET|"
                          r"GRAB&GO|new|2026)\b", re.I)


def dish_name(title: str) -> str:
    """The dish, without the service codes a menu system hangs off it."""
    head = title.split(" - ")[0]
    return re.sub(r"\s+", " ", _TITLE_NOISE.sub(" ", head)).strip(" ,-")


def query_text(card, resolved, top_n: int = 5) -> str:
    """What to ask the corpus: the dish plus its most distinctive ingredients."""
    names = [r.canonical.replace("_", " ") for r in resolved if r.quantified]
    return " ".join([dish_name(card.title)] + names[:top_n]).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path,
                    default=Path("../tests/Recipe_Cards_Merged_v001_abstract.pdf"))
    ap.add_argument("--domain", default="canon_library")
    ap.add_argument("--cache", type=Path,
                    default=Path("../evidence/T101/corpus-index.json"))
    ap.add_argument("--json", type=Path,
                    default=Path("../evidence/T101/validate-cards.json"))
    ap.add_argument("--pool", type=int, default=25,
                    help="quante pagine il recupero propone prima del riordino")
    ap.add_argument("--explain", type=int, default=0,
                    help="stampa la spiegazione delle prime N schede")
    args = ap.parse_args()

    reg = Registry.load()
    cards = load_cards(args.cards)
    loader = Neo4jLoader()
    index = CorpusIndex.from_graph(loader, args.domain, registry=reg, cache=args.cache)
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    embedder.encode(["warm up"])
    enc = lambda q: embedder.encode([q])[0]  # noqa: E731

    rows = []
    try:
        for card in cards:
            resolved = card.resolved(reg)
            query = RecipeQuery(title=dish_name(card.title), resolved=resolved,
                                proportions=proportions(resolved),
                                verbs=techniques_in(card.procedure),
                                verb_seq=techniques_in(card.procedure, ordered=True))

            t0 = time.perf_counter()
            fp = index.rank(query)[:5]
            fp_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            text = query_text(card, resolved)
            res = retriever.query(text, embed_fn=enc, domain=args.domain,
                                  channels=("vector", "fulltext", "graph"),
                                  hops=1, result_window=args.pool)
            props = retriever.hydrate([h.node_id for h in res.hits[: args.pool]])
            pool = []
            for nid in [h.node_id for h in res.hits[: args.pool]]:
                p = props.get(nid) or {}
                if (p.get("extraction_method") or "") != "page":
                    continue
                # `hydrate` returns provenance as text ("pages 279-279") and
                # not as page_lo: reading a field it does not send back left
                # every pool empty and every card refused, which looked like a
                # result and was a typo.
                m = re.search(r"(\d+)", p.get("source_location") or "")
                if not m:
                    continue
                cand = index.page(p.get("source_file") or "", int(m.group(1)))
                if cand is not None:
                    pool.append(cand)
            rr = index.rank(query, candidates=pool)[:5] if pool else []
            rr_ms = (time.perf_counter() - t0) * 1000

            conf = confidence(rr or fp, query, index.idf)
            supported = bool(rr) and not res.refused
            rows.append({
                "number": card.number, "title": card.title[:70],
                "dish": dish_name(card.title),
                "lines": len(card.lines), "quantified": query and len(query.proportions),
                "retrieval_pool": len(pool),
                "refused": bool(res.refused) or not pool,
                "verdict": "SUPPORTED" if supported else "NOT_SUPPORTED",
                "retrieval_first": [m.as_dict() for m in rr[:3]],
                "fingerprint_first": [m.as_dict() for m in fp[:3]],
                "agree": bool(rr and fp
                              and rr[0].candidate.source_file == fp[0].candidate.source_file
                              and rr[0].candidate.page == fp[0].candidate.page),
                "confidence": conf,
                "ms": {"fingerprint": round(fp_ms, 1), "retrieval": round(rr_ms, 1)},
            })
            if args.explain and card.number <= args.explain and rr:
                print(f"\n### {card.title[:70]}")
                print(explain(query, rr[0], index.idf))
    finally:
        loader.close()

    answered = [r for r in rows if r["verdict"] == "SUPPORTED"]
    report = {
        "cards": len(rows),
        "source": str(args.cards),
        "ground_truth": "NONE — this export declares no reference work; these "
                        "verdicts are proposals for human validation, not scores",
        "answered": len(answered),
        "refused": sum(1 for r in rows if r["refused"]),
        "methods_agree_pct": round(100 * sum(1 for r in answered if r["agree"])
                                   / max(len(answered), 1), 1),
        "retrieval_pool_mean": round(statistics.mean(r["retrieval_pool"] for r in rows), 1),
        "confidence_margin_median": round(statistics.median(
            r["confidence"]["margin"] for r in answered), 4) if answered else None,
        "latency_ms": {
            "fingerprint_mean": round(statistics.mean(r["ms"]["fingerprint"] for r in rows), 1),
            "retrieval_mean": round(statistics.mean(r["ms"]["retrieval"] for r in rows), 1),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"report": report, "rows": rows},
                                    indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
