#!/usr/bin/env python3
"""Phase 6.7 — plant known contradictions and supersessions, then measure detection.

Acceptance (execution plan §6): ≥ 8/10 planted contradictions detected; zero
silent resolutions; planted supersessions ordered by doc_date.

The planted pairs are synthetic documents built from culinary facts of the same
shape as the corpus (temperatures, ratios, times) so that blocking's cosine
signal behaves as it does on real content: the two sides of a planted pair are
*about* the same thing and differ in the claim.

`--adjudicator rules` uses a deterministic stand-in that reads the planted
numbers — this measures the blocking + resolution machinery honestly without
LLM credentials. Swap in the real batched adjudicator when a key is available;
the interface is identical.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from graphify_ent.clash import ClashEngine, ClashPair, Verdict, VerdictCache
from graphify_ent.loader import Neo4jLoader

DOMAIN = "clashtest"

# 10 contradictions: same subject, incompatible claim.
CONTRADICTIONS = [
    ("beef stock simmering time", "Simmer beef stock for 8 hours.", "Simmer beef stock for 3 hours."),
    ("chocolate tempering temperature", "Temper dark chocolate at 31 degrees Celsius.",
     "Temper dark chocolate at 45 degrees Celsius."),
    ("bechamel butter flour ratio", "Use equal parts butter and flour for bechamel.",
     "Use twice as much butter as flour for bechamel."),
    ("bread proofing time", "Proof the dough for 90 minutes at room temperature.",
     "Proof the dough for 20 minutes at room temperature."),
    ("fish storage temperature", "Store fresh fish at 0 degrees Celsius.",
     "Store fresh fish at 8 degrees Celsius."),
    ("pasta water salt", "Add 10 grams of salt per litre of pasta water.",
     "Add 40 grams of salt per litre of pasta water."),
    ("egg custard setting point", "Custard sets at 82 degrees Celsius.",
     "Custard sets at 96 degrees Celsius."),
    ("roast chicken oven temperature", "Roast chicken at 180 degrees Celsius.",
     "Roast chicken at 250 degrees Celsius."),
    ("mayonnaise oil ratio", "Use 200 ml of oil per egg yolk for mayonnaise.",
     "Use 20 ml of oil per egg yolk for mayonnaise."),
    ("caramel cooking stage", "Cook caramel to 170 degrees Celsius.",
     "Cook caramel to 120 degrees Celsius."),
]

# 5 supersessions: v1/v2 of the same policy, newer doc_date must win.
SUPERSESSIONS = [
    ("cold storage policy", "Cold storage is set to 4 degrees Celsius.",
     "Cold storage is set to 2 degrees Celsius."),
    ("allergen labelling policy", "Allergens are listed in the appendix.",
     "Allergens are listed on the front label."),
    ("supplier audit frequency", "Suppliers are audited annually.",
     "Suppliers are audited twice per year."),
    ("waste disposal procedure", "Organic waste is collected weekly.",
     "Organic waste is collected daily."),
    ("kitchen cleaning schedule", "Deep cleaning happens monthly.",
     "Deep cleaning happens fortnightly."),
]


def build_nodes() -> tuple[list[dict], list[dict], dict]:
    nodes, truth = [], {"contradictions": [], "supersessions": []}

    for i, (subject, claim_a, claim_b) in enumerate(CONTRADICTIONS):
        a, b = f"clash_c{i}_a", f"clash_c{i}_b"
        for nid, claim, src in ((a, claim_a, f"handbook_A_{i}.pdf"),
                                (b, claim_b, f"handbook_B_{i}.pdf")):
            nodes.append({
                "id": nid, "label": subject, "label_en": subject, "lang": "en",
                "file_type": "document", "source_file": src,
                "text_excerpt": f"{subject}. {claim}", "evidence": claim,
                "confidence": "EXTRACTED", "extraction_method": "native",
                "source_rank": 2, "doc_date": "2026-01-01", "doc_date_confidence": 0.9,
            })
        truth["contradictions"].append([a, b])

    for i, (subject, old_claim, new_claim) in enumerate(SUPERSESSIONS):
        a, b = f"clash_s{i}_v1", f"clash_s{i}_v2"
        nodes.append({
            "id": a, "label": subject, "label_en": subject, "lang": "en",
            "file_type": "document", "source_file": f"policy_{i}_v1.pdf", "version": 1,
            "text_excerpt": f"{subject}. {old_claim}", "evidence": old_claim,
            "confidence": "EXTRACTED", "extraction_method": "native",
            "source_rank": 2, "doc_date": "2024-01-01", "doc_date_confidence": 0.9,
        })
        nodes.append({
            "id": b, "label": subject, "label_en": subject, "lang": "en",
            "file_type": "document", "source_file": f"policy_{i}_v2.pdf", "version": 2,
            "text_excerpt": f"{subject}. {new_claim}", "evidence": new_claim,
            "confidence": "EXTRACTED", "extraction_method": "native",
            "source_rank": 2, "doc_date": "2026-06-01", "doc_date_confidence": 0.9,
        })
        truth["supersessions"].append([a, b])

    return nodes, [], truth


_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def rules_adjudicator(pairs):
    """Deterministic stand-in: two claims about one subject with different
    numbers contradict. Quotes both excerpts, as the real prompt must."""
    out = []
    for p in pairs:
        na, nb = _NUM.findall(p.a_text), _NUM.findall(p.b_text)
        contradictory = bool(na) and bool(nb) and na != nb
        out.append(Verdict(
            verdict="CONTRADICTORY" if contradictory else "COMPLEMENTARY",
            rationale=(f'"{p.a_text[:80]}" vs "{p.b_text[:80]}"'),
            confidence=0.9 if contradictory else 0.6,
            a_excerpt=p.a_text[:120], b_excerpt=p.b_text[:120],
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--cosine", type=float, default=0.85)
    ap.add_argument("--apply", action="store_true",
                    help="write edges (auto-resolution stays gated on G4)")
    args = ap.parse_args()

    nodes, edges, truth = build_nodes()
    tmp = Path("/tmp/planted-clashes.json")
    tmp.write_text(json.dumps({"nodes": nodes, "edges": edges}))

    loader = Neo4jLoader()
    try:
        with loader._session() as s:
            s.run(f"MATCH (n:Entity {{domain:'{DOMAIN}'}}) DETACH DELETE n")
        loader.apply_schema()
        loader.load(tmp, domain=DOMAIN)

        from graphify_ent.embed import embed_graph
        embed_graph(loader, progress=False)

        engine = ClashEngine(loader, cache=VerdictCache(Path("/tmp/verdicts-planted.json")))
        pairs = engine.block_pairs(domain=DOMAIN, cosine=args.cosine)
        checkpoint = engine.blocking_checkpoint(pairs, domain=DOMAIN)

        judged = engine.adjudicate(pairs, rules_adjudicator)
        resolutions = engine.apply_resolutions(judged, dry_run=not args.apply)

        found = {tuple(sorted((p.a_id, p.b_id))) for p, v in judged
                 if v.verdict == "CONTRADICTORY"}
        planted_c = [tuple(sorted(x)) for x in truth["contradictions"]]
        detected = [p for p in planted_c if p in found]

        # Supersessions: version-level engine decides by version + doc_date.
        from graphify_ent.temporal import TemporalEngine
        trep = TemporalEngine(loader).apply_version_supersession(domain=DOMAIN) \
            if args.apply else None

        ordered_ok = 0
        if args.apply:
            with loader._session() as s:
                for old, new in truth["supersessions"]:
                    rec = s.run(
                        "MATCH (o:Entity {id:$o})-[r:SUPERSEDED_BY]->(n:Entity {id:$n}) "
                        "RETURN o.valid_to AS vt", o=old, n=new).single()
                    if rec and rec["vt"] == "2026-06-01":
                        ordered_ok += 1

        silent = [r for r in resolutions
                  if r.action == "supersede" and (not r.policy or not r.rationale)]

        report = {
            "blocking": checkpoint,
            "planted_contradictions": len(planted_c),
            "detected_contradictions": len(detected),
            "detection_rate": round(100 * len(detected) / len(planted_c), 1),
            "acceptance_8_of_10": len(detected) >= 8,
            "planted_supersessions": len(truth["supersessions"]),
            "supersessions_correctly_ordered": ordered_ok if args.apply else "not applied (G4 gate)",
            "silent_resolutions": len(silent),
            "zero_silent_resolutions": len(silent) == 0,
            "verdict_cache_hits": engine.cache.hits,
            "resolution_policies": {},
        }
        for r in resolutions:
            if r.action == "supersede":
                report["resolution_policies"][r.policy] = \
                    report["resolution_policies"].get(r.policy, 0) + 1

        print(json.dumps(report, indent=2))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2))
        if args.report:
            args.report.write_text(engine.conflict_report(resolutions, run_id="planted-eval"))
        return 0
    finally:
        loader.close()


if __name__ == "__main__":
    raise SystemExit(main())
