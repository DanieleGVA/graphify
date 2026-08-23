#!/usr/bin/env python3
"""Real semantic extraction over the whole pilot corpus.

Until now the loaded graph came from structural extraction — a stand-in. This
runs the actual LLM extraction the architecture specifies, over every slice of
every book, with the compact schema and the prose relation vocabulary.

Design notes:

  * **Resumable.** 767 slices is long enough that a crash, a rate limit or a
    laptop lid must not cost the whole run. Every finished slice is appended to
    a JSONL checkpoint and skipped on restart.

  * **Fail-closed at ingest.** A node whose `evidence` is not a literal
    substring of its own slice never enters the graph. Q1 faithfulness is a
    hard gate; enforcing it at write time means it cannot be violated at read
    time. Edges whose endpoints did not survive are dropped with them.

  * **Ids are namespaced per (book, slice).** The model reuses obvious ids
    ("consomme") across slices, and MERGE on a bare id would fuse nodes whose
    evidence quotes come from different pages — silently breaking the very
    binding the gate checks. Cross-document unification is the :Concept
    layer's job, not the loader's.

  * **Concurrency is bounded and retried.** Work is network-bound, so threads
    are the right tool; failures back off and retry rather than losing a slice.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from graphify_ent.compact_schema import COMPACT_EXTRACTION_SYSTEM, parse_compact
from graphify_ent.file_slice import read_slice_text, slice_pdf

CHAR_CAP = 20_000
PRICING = {"deepseek-v4-flash": {"input": 0.14, "output": 0.28}}
_ID_OK = re.compile(r"[^a-z0-9_]+")

_print_lock = threading.Lock()
_ckpt_lock = threading.Lock()


def norm(s) -> str:
    """Whitespace-normalise. Tolerates the model returning a list for a field
    the schema declares as a string — measured once in 767 slices, and losing a
    whole slice to it would be a worse failure than joining the parts."""
    if isinstance(s, (list, tuple)):
        s = " ".join(str(x) for x in s)
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def slug(name: str, n: int = 22) -> str:
    return _ID_OK.sub("_", name.lower())[:n].strip("_")


def call(model: str, system: str, user: str, timeout: int, attempts: int = 3):
    last = None
    for a in range(attempts):
        try:
            payload = json.dumps({
                "model": model, "stream": False, "think": False,
                "options": {"temperature": 0, "num_ctx": 32768, "num_predict": 16000},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }).encode()
            req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                env = json.loads(r.read())
            return (env.get("message") or {}).get("content") or "", {
                "in_tokens": env.get("prompt_eval_count", 0),
                "out_tokens": env.get("eval_count", 0),
                "truncated": env.get("done_reason") == "length",
            }
        except Exception as exc:                      # network, timeout, 5xx
            last = exc
            time.sleep(2 ** a * 3)
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:120]}")


def do_slice(task: dict, model: str, timeout: int) -> dict:
    pdf = Path(task["pdf"])
    sl = task["slice_obj"]
    text = read_slice_text(sl)
    user = (f"=== {pdf.name} (pages {sl.page_start+1}-{sl.page_end+1}) ===\n{text}")
    t0 = time.perf_counter()
    content, meta = call(model, COMPACT_EXTRACTION_SYSTEM, user, timeout)
    nodes, edges = parse_compact(content)

    src = norm(text)
    pref = f"{slug(pdf.stem)}_s{task['index']}"
    kept, idmap, by_id = [], {}, {}
    for n in nodes:
        ev = n.get("evidence") or ""
        if isinstance(ev, (list, tuple)):
            ev = " ".join(str(x) for x in ev)
            n["evidence"] = ev
        if not norm(ev) or norm(ev) not in src:       # fail-closed at ingest
            continue
        raw = str(n.get("id") or "").strip().lower()
        if not raw:
            continue
        nid = f"{pref}_{_ID_OK.sub('_', raw)}"[:120]
        idmap[raw] = nid
        n["id"] = nid
        n["source_file"] = pdf.name
        n["extraction_method"] = "llm_deepseek"
        n.setdefault("source_location", f"pages {sl.page_start+1}-{sl.page_end+1}")
        # The model routinely emits the same entity several times in one slice,
        # each with a different supporting quote. Keep one node, and keep the
        # LONGEST verified quote: a longer excerpt is a stronger anchor, and
        # dropping later duplicates blindly would discard the better evidence.
        prev = by_id.get(nid)
        if prev is None:
            by_id[nid] = n
            kept.append(n)
        elif len(ev) > len(prev.get("evidence") or ""):
            prev.update(n)

    ke = []
    for e in edges:
        s_, t_ = idmap.get(str(e.get("source", "")).lower()), idmap.get(
            str(e.get("target", "")).lower())
        if not s_ or not t_ or s_ == t_:
            continue
        e["source"], e["target"] = s_, t_
        e["source_file"] = pdf.name
        ke.append(e)

    return {
        "book": pdf.name, "index": task["index"],
        "pages": [sl.page_start + 1, sl.page_end + 1],
        "chars": len(text), "nodes": kept, "edges": ke,
        "nodes_returned": len(nodes), "nodes_kept": len(kept),
        "edges_kept": len(ke), "wall_s": round(time.perf_counter() - t0, 1), **meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("../pilot"))
    ap.add_argument("--model", default="deepseek-v4-flash:cloud")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N slices")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("../evidence/T71/slices.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("../evidence/T71/corpus-graph.json"))
    ap.add_argument("--stats", type=Path, default=Path("../evidence/T71/ingest-stats.json"))
    args = ap.parse_args()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for pdf in sorted(args.corpus.glob("*.pdf")):
        for i, sl in enumerate(slice_pdf(pdf, CHAR_CAP)):
            tasks.append({"pdf": str(pdf), "index": i, "slice_obj": sl,
                          "key": f"{pdf.name}#{i}"})
    if args.limit:
        tasks = tasks[: args.limit]

    done: dict[str, dict] = {}
    if args.checkpoint.exists():
        for line in args.checkpoint.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[f"{rec['book']}#{rec['index']}"] = rec
    todo = [t for t in tasks if t["key"] not in done]
    print(f"porzioni totali {len(tasks)}  già fatte {len(done)}  da fare {len(todo)}",
          flush=True)

    t_start = time.perf_counter()
    failures: list[dict] = []
    completed = 0
    fh = args.checkpoint.open("a")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(do_slice, t, args.model, args.timeout): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                failures.append({"key": t["key"], "error": str(exc)[:200]})
                with _print_lock:
                    print(f"  FALLITA {t['key']}: {str(exc)[:90]}", flush=True)
                continue
            with _ckpt_lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[t["key"]] = rec
            completed += 1
            if completed % 10 == 0 or completed == len(todo):
                el = time.perf_counter() - t_start
                rate = completed / el
                eta = (len(todo) - completed) / rate / 60 if rate else 0
                kept = sum(r["nodes_kept"] for r in done.values())
                with _print_lock:
                    print(f"  {completed}/{len(todo)}  nodi {kept:,}  "
                          f"{rate*60:.1f} porzioni/min  restano ~{eta:.0f} min", flush=True)
    fh.close()

    # --- assemble -------------------------------------------------------
    nodes, edges = [], []
    seen = set()
    for rec in done.values():
        for n in rec["nodes"]:
            if n["id"] in seen:
                continue
            seen.add(n["id"])
            nodes.append(n)
        edges.extend(rec["edges"])

    ret = sum(r["nodes_returned"] for r in done.values())
    out_tok = sum(r.get("out_tokens", 0) for r in done.values())
    in_tok = sum(r.get("in_tokens", 0) for r in done.values())
    p = PRICING.get(args.model.split(":")[0])
    stats = {
        "model": args.model, "slices_total": len(tasks), "slices_done": len(done),
        "slices_failed": len(failures), "failures": failures[:20],
        "truncated": sum(1 for r in done.values() if r.get("truncated")),
        "nodes_returned": ret, "nodes_kept": len(nodes),
        "evidence_verified_pct": round(100 * len(nodes) / ret, 2) if ret else 0.0,
        "edges": len(edges), "chars": sum(r["chars"] for r in done.values()),
        "in_tokens": in_tok, "out_tokens": out_tok,
        "cost_usd": round(in_tok / 1e6 * p["input"] + out_tok / 1e6 * p["output"], 4)
        if p else None,
        "wall_minutes": round((time.perf_counter() - t_start) / 60, 1),
    }
    args.out.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False))
    args.stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\ngrafo -> {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
