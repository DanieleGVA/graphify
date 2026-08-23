#!/usr/bin/env python3
"""Run the REAL LLM semantic extraction pass over one document.

This is the component the structural extractor stood in for. It routes through
the locally-installed Claude Code CLI (`claude -p`), so it uses the operator's
existing subscription rather than a pay-as-you-go API key — the `claude-cli`
backend upstream already declares (graphify/llm.py).

Per slice it sends the ENTERPRIPHY extraction prompt (which mandates
`label_orig` / `label_en` / `lang` and an evidence excerpt per node) plus the
slice text, then validates the result:

  * evidence must be a literal substring of the dispatched slice — a node whose
    quote cannot be found in the source is dropped and counted, because that is
    a fabrication and Q1 treats it as a defect, not a low score;
  * label_en is normalized host-side so the model's phrasing converges.

Usage:
  python tools/extract_semantic.py <pdf> --max-slices 2 --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from graphify_ent.file_slice import read_slice_text, slice_pdf
from graphify_ent.labels import ENT_EXTRACTION_SYSTEM, coverage_report, enrich_nodes

CHAR_CAP = 20_000


def call_claude(user_message: str, timeout: int = 600) -> tuple[dict, dict]:
    """Invoke the local Claude Code CLI. Returns (parsed_extraction, usage)."""
    proc = subprocess.run(
        ["claude", "-p", "--output-format", "json", "--no-session-persistence",
         "--append-system-prompt", ENT_EXTRACTION_SYSTEM],
        input=user_message, capture_output=True, text=True, encoding="utf-8",
        timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p exited {proc.returncode}: {proc.stderr.strip()[:300]}")

    envelope = json.loads(proc.stdout)
    raw = envelope.get("result") or ""
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(m.group(0)) if m else {"nodes": [], "edges": []}
    return parsed, {
        "cost_usd": envelope.get("total_cost_usd", 0.0),
        "duration_ms": envelope.get("duration_api_ms", 0),
        "usage": envelope.get("usage", {}),
    }


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--max-slices", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--domain", default="pilot")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    slices = slice_pdf(args.pdf, CHAR_CAP)[args.start : args.start + args.max_slices]
    print(f"{args.pdf.name}: dispatching {len(slices)} slices of {CHAR_CAP:,} chars\n")

    all_nodes, all_edges = [], []
    total_cost = 0.0
    fabricated = 0
    per_slice = []

    for i, sl in enumerate(slices, 1):
        text = read_slice_text(sl)
        header = f"=== {args.pdf.name} (pages {sl.page_start+1}-{sl.page_end+1}) ===\n"
        t0 = time.perf_counter()
        try:
            parsed, meta = call_claude(header + text)
        except Exception as exc:
            print(f"  slice {i}: FAILED {exc}")
            per_slice.append({"slice": i, "error": str(exc)})
            continue
        wall = time.perf_counter() - t0
        total_cost += meta["cost_usd"]

        nodes = parsed.get("nodes") or []
        edges = parsed.get("edges") or []

        # Evidence binding: the quote must exist verbatim in what we sent.
        src = norm(text)
        kept = []
        for n in nodes:
            ev = norm(n.get("evidence", ""))
            if ev and ev in src:
                n.setdefault("source_file", args.pdf.name)
                n["source_location"] = f"pages {sl.page_start+1}-{sl.page_end+1}"
                n["text_excerpt"] = n.get("evidence", "")[:1000]
                n["extraction_method"] = "llm"
                kept.append(n)
            else:
                fabricated += 1
        kept = enrich_nodes(kept, default_lang=None)
        kept_ids = {n.get("id") for n in kept}
        kept_edges = [e for e in edges
                      if e.get("source") in kept_ids and e.get("target") in kept_ids]

        all_nodes.extend(kept)
        all_edges.extend(kept_edges)
        cov = coverage_report(kept)
        per_slice.append({
            "slice": i, "pages": f"{sl.page_start+1}-{sl.page_end+1}",
            "nodes_returned": len(nodes), "nodes_kept": len(kept),
            "edges_returned": len(edges), "edges_kept": len(kept_edges),
            "label_en_pct": cov["coverage_pct"],
            "cost_usd": round(meta["cost_usd"], 4), "wall_s": round(wall, 1),
        })
        print(f"  slice {i} (pp {sl.page_start+1}-{sl.page_end+1}): "
              f"{len(kept)}/{len(nodes)} nodes kept, {len(kept_edges)} edges, "
              f"label_en {cov['coverage_pct']}%, ${meta['cost_usd']:.3f}, {wall:.0f}s")

    cov = coverage_report(all_nodes)
    rels = {}
    for e in all_edges:
        rels[e.get("relation", "?")] = rels.get(e.get("relation", "?"), 0) + 1
    langs = {}
    for n in all_nodes:
        langs[n.get("lang")] = langs.get(n.get("lang"), 0) + 1

    report = {
        "pdf": args.pdf.name,
        "slices_dispatched": len(slices),
        "nodes": len(all_nodes),
        "edges": len(all_edges),
        "fabricated_dropped": fabricated,
        "label_en_coverage_pct": cov["coverage_pct"],
        "lang_distribution": langs,
        "relation_types": rels,
        "total_cost_usd": round(total_cost, 4),
        "cost_per_slice_usd": round(total_cost / max(len(slices), 1), 4),
        "per_slice": per_slice,
    }
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "per_slice"},
                            indent=2, ensure_ascii=False))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        graph_out = args.json.with_name(args.json.stem + "-graph.json")
        graph_out.write_text(json.dumps({"nodes": all_nodes, "edges": all_edges},
                                        indent=2, ensure_ascii=False))
        print(f"\ngraph -> {graph_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
