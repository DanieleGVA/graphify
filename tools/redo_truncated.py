#!/usr/bin/env python3
"""Re-extract the slices whose response was cut off mid-answer.

145 of 767 slices hit the output cap. They are not empty — they carry 33,361
nodes — but the response was truncated, and the schema puts edges AFTER nodes,
so what was lost is disproportionately the relations.

Raising the cap is not the fix: the cap is the model's, and a longer answer on
the same input truncates in the same place. Halving the input is. Each
truncated slice is re-run as two half-page-range sub-slices, so each response
has half as much to say and fits. A sub-slice that truncates again is split
again, down to a single page.

Sub-slice results replace their parent at assembly time; the original record is
kept in the checkpoint so the run stays auditable.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ingest_corpus import PRICING, do_slice

from graphify_ent.file_slice import PdfSlice

_lock = threading.Lock()
MAX_DEPTH = 3


def split(sl: PdfSlice) -> list[PdfSlice]:
    """Halve a page range. A single page cannot be split further."""
    if sl.page_end <= sl.page_start:
        return []
    mid = (sl.page_start + sl.page_end) // 2
    return [PdfSlice(sl.path, sl.page_start, mid, 0),
            PdfSlice(sl.path, mid + 1, sl.page_end, 0)]


def redo(rec: dict, model: str, timeout: int, depth: int = 0) -> list[dict]:
    """Re-extract one truncated record, splitting until it fits or runs out."""
    pdf = Path(rec["pdf_path"])
    parent = PdfSlice(pdf, rec["pages"][0] - 1, rec["pages"][1] - 1, 0)
    parts = split(parent)
    if not parts:
        return []
    out = []
    for j, sub in enumerate(parts):
        task = {"pdf": str(pdf), "index": f"{rec['index']}_{depth}{j}", "slice_obj": sub}
        try:
            sub_rec = do_slice(task, model, timeout)
        except Exception as exc:
            with _lock:
                print(f"    sotto-porzione {task['index']} fallita: {str(exc)[:70]}",
                      flush=True)
            continue
        sub_rec["parent_index"] = rec["index"]
        sub_rec["pdf_path"] = str(pdf)
        if sub_rec.get("truncated") and depth + 1 < MAX_DEPTH:
            deeper = redo(sub_rec, model, timeout, depth + 1)
            out.extend(deeper if deeper else [sub_rec])
        else:
            out.append(sub_rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("../pilot"))
    ap.add_argument("--model", default="deepseek-v4-flash:cloud")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke test")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("../evidence/T71/slices.jsonl"))
    ap.add_argument("--redo-checkpoint", type=Path,
                    default=Path("../evidence/T71/slices-redo.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("../evidence/T71/corpus-graph.json"))
    ap.add_argument("--stats", type=Path, default=Path("../evidence/T71/redo-stats.json"))
    args = ap.parse_args()

    by_name = {p.name: p for p in args.corpus.glob("*.pdf")}
    records = [json.loads(l) for l in args.checkpoint.read_text().splitlines() if l.strip()]
    for r in records:
        r["pdf_path"] = str(by_name.get(r["book"], args.corpus / r["book"]))

    truncated = [r for r in records if r.get("truncated")]
    already = {}
    if args.redo_checkpoint.exists():
        for line in args.redo_checkpoint.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                already.setdefault(rec["parent_index_key"], []).append(rec)

    todo = [r for r in truncated
            if f"{r['book']}#{r['index']}" not in already]
    if args.limit:
        todo = todo[: args.limit]
    print(f"troncate {len(truncated)}  già rifatte {len(already)}  da rifare {len(todo)}",
          flush=True)

    t0 = time.perf_counter()
    fh = args.redo_checkpoint.open("a")
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(redo, r, args.model, args.timeout): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            key = f"{r['book']}#{r['index']}"
            try:
                subs = fut.result()
            except Exception as exc:
                with _lock:
                    print(f"  FALLITA {key}: {str(exc)[:80]}", flush=True)
                continue
            with _lock:
                for s in subs:
                    s["parent_index_key"] = key
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                fh.flush()
                already[key] = subs
            completed += 1
            if completed % 10 == 0 or completed == len(todo):
                el = time.perf_counter() - t0
                eta = (len(todo) - completed) / (completed / el) / 60 if completed else 0
                with _lock:
                    print(f"  {completed}/{len(todo)}  restano ~{eta:.0f} min", flush=True)
    fh.close()

    # --- reassemble: sub-slices replace their truncated parent ------------
    nodes, edges, seen = [], [], set()
    replaced = 0
    for r in records:
        key = f"{r['book']}#{r['index']}"
        source = already.get(key)
        if source:
            replaced += 1
        for rec in (source if source else [r]):
            for n in rec["nodes"]:
                if n["id"] in seen:
                    continue
                seen.add(n["id"])
                nodes.append(n)
            edges.extend(rec["edges"])

    before_nodes = sum(len(r["nodes"]) for r in records)
    before_edges = sum(len(r["edges"]) for r in records)
    sub_recs = [s for v in already.values() for s in v]
    out_tok = sum(s.get("out_tokens", 0) for s in sub_recs)
    in_tok = sum(s.get("in_tokens", 0) for s in sub_recs)
    p = PRICING.get(args.model.split(":")[0])
    stats = {
        "truncated_parents": len(truncated), "parents_replaced": replaced,
        "sub_slices_run": len(sub_recs),
        "sub_slices_still_truncated": sum(1 for s in sub_recs if s.get("truncated")),
        "nodes_before": before_nodes, "nodes_after": len(nodes),
        "edges_before": before_edges, "edges_after": len(edges),
        "extra_cost_usd": round(in_tok / 1e6 * p["input"] + out_tok / 1e6 * p["output"], 4)
        if p else None,
        "wall_minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    args.out.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False))
    args.stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
