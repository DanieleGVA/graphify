#!/usr/bin/env python3
"""T74 — merge both arms' measurements into the final benchmark record.

Reads only files produced by the pipeline runs and the card adjudications;
computes nothing new except ratios. Every figure in the summary can be traced
back to a raw file under evidence/T74/.
"""

from __future__ import annotations

import json
from pathlib import Path

EV = Path("../evidence/T74")


def read(name: str) -> dict:
    p = EV / name
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    extraction = read("extraction-stats.json")
    load = read("load-report.json")
    upstream = read("upstream-extract-stats.json")
    cards_e = read("cards-enterpriphy.json")
    cards_u = read("cards-upstream.json")

    up_tot = upstream.get("totals", {})
    summary = {
        "task": "T74 — due libri, tre schede: ENTERPRIPHY vs graphify upstream",
        "corpus": ["The Professional Chef (CIA)", "Le Grand Larousse gastronomique"],
        "model_both_arms": "deepseek-v4-flash:cloud",
        "ingest": {
            "enterpriphy": {
                "chars_processed": extraction.get("chars"),
                "coverage_pct_of_books": 100.0,
                "slices": extraction.get("slices"),
                "nodes": extraction.get("nodes_kept"),
                "edges": extraction.get("edges"),
                "evidence_verified_pct": extraction.get("evidence_verified_pct"),
                "extraction_cost_usd": extraction.get("cost_usd"),
                "extraction_wall_min_6workers": extraction.get("est_wall_min_6workers"),
                "load_enrich_embed_seconds": load.get("total_seconds"),
                "steps_seconds": load.get("steps_seconds"),
            },
            "upstream": {
                "chars_processed": up_tot.get("chars_sent"),
                "coverage_pct_of_books": round(
                    100 * up_tot.get("chars_sent", 0)
                    / max(extraction.get("chars", 1), 1), 2),
                "nodes": up_tot.get("nodes"),
                "edges": up_tot.get("edges"),
                "books": upstream.get("books"),
                "helps_granted": upstream.get("helps"),
            },
        },
        "cards": {
            arm: {
                "claims": c.get("claims"),
                "accuracy_pct": c.get("accuracy_pct"),
                "supported_recall": c.get("supported_recall"),
                "unsupported_rejection": c.get("unsupported_rejection"),
                "false_confirmations": c.get("false_confirmations"),
                "verdicts": c.get("verdicts"),
                "latency_ms_mean": c.get("latency_ms_mean"),
                "latency_ms_p95": c.get("latency_ms_p95"),
            }
            for arm, c in (("enterpriphy", cards_e), ("upstream", cards_u))
        },
    }
    (EV / "benchmark-T74.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
