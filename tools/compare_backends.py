#!/usr/bin/env python3
"""Compare extraction backends on the SAME slice, by cost per *verified* node.

Cost per call is the wrong metric: a cheap model that fabricates is worthless,
because every node whose quote cannot be found verbatim in the source is
dropped by the Q1 evidence gate. What matters is:

    cost per node that survives the evidence check

plus label_en coverage (Phase 1.5) and the share of fabrications, which is a
direct read on how much the model invents under this prompt.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from graphify_ent.file_slice import read_slice_text, slice_pdf
from graphify_ent.compact_schema import COMPACT_EXTRACTION_SYSTEM, parse_compact
from graphify_ent.labels import ENT_EXTRACTION_SYSTEM, coverage_report, enrich_nodes

CHAR_CAP = 20_000
# Per-million-token prices as declared in graphify/llm.py BACKENDS.
PRICING = {
    "claude-cli": {"input": 3.0, "output": 15.0, "note": "billed to the Claude plan"},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28, "note": "via ollama cloud"},
    "qwen3": {"input": 0.0, "output": 0.0, "note": "local"},
    "gemma3": {"input": 0.0, "output": 0.0, "note": "local"},
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def parse_json_blob(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}


def run_claude_cli(prompt: str, timeout: int) -> tuple[dict, dict]:
    proc = subprocess.run(
        ["claude", "-p", "--output-format", "json", "--no-session-persistence",
         "--append-system-prompt", ENT_EXTRACTION_SYSTEM],
        input=prompt, capture_output=True, text=True, encoding="utf-8",
        timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:300])
    env = json.loads(proc.stdout)
    u = env.get("usage", {}) or {}
    return parse_json_blob(env.get("result") or ""), {
        "cost_usd": env.get("total_cost_usd", 0.0),
        "in_tokens": (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                      + u.get("cache_creation_input_tokens", 0)),
        "out_tokens": u.get("output_tokens", 0),
    }


def run_ollama(model: str, prompt: str, timeout: int, compact: bool = True) -> tuple[dict, dict]:
    import urllib.request

    system = COMPACT_EXTRACTION_SYSTEM if compact else ENT_EXTRACTION_SYSTEM
    payload = json.dumps({
        "model": model, "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 32768, "num_predict": 16000},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        env = json.loads(resp.read())
    content = (env.get("message") or {}).get("content") or ""
    in_tok = env.get("prompt_eval_count", 0)
    if compact:
        n, e = parse_compact(content)
        parsed_compact = {"nodes": n, "edges": e}
    out_tok = env.get("eval_count", 0)
    base = model.split(":")[0]
    price = PRICING.get(base, {"input": 0.0, "output": 0.0})
    cost = in_tok / 1e6 * price["input"] + out_tok / 1e6 * price["output"]
    result = parsed_compact if compact else parse_json_blob(content)
    return result, {"cost_usd": cost, "in_tokens": in_tok, "out_tokens": out_tok,
                    "truncated": env.get("done_reason") == "length"}


def evaluate(parsed: dict, source: str) -> dict:
    nodes = parsed.get("nodes") or []
    edges = parsed.get("edges") or []
    src = norm(source)
    kept, fabricated, no_evidence = [], 0, 0
    for n in nodes:
        ev = norm(n.get("evidence", ""))
        if not ev:
            no_evidence += 1
            continue
        if ev in src:
            kept.append(n)
        else:
            fabricated += 1
    kept = enrich_nodes(kept, default_lang=None)
    kept_ids = {n.get("id") for n in kept}
    kept_edges = [e for e in edges
                  if e.get("source") in kept_ids and e.get("target") in kept_ids]
    cov = coverage_report(kept)
    return {
        "nodes_returned": len(nodes), "nodes_verified": len(kept),
        "fabricated": fabricated, "no_evidence": no_evidence,
        "edges_returned": len(edges), "edges_kept": len(kept_edges),
        "relation_types": len({e.get("relation") for e in kept_edges}),
        "label_en_pct": cov["coverage_pct"],
        "verified_rate_pct": round(100 * len(kept) / len(nodes), 1) if nodes else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--slice-index", type=int, default=20)
    ap.add_argument("--backends", default="deepseek-v4-flash:cloud")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    sl = slice_pdf(args.pdf, CHAR_CAP)[args.slice_index]
    text = read_slice_text(sl)
    prompt = f"=== {args.pdf.name} (pages {sl.page_start+1}-{sl.page_end+1}) ===\n{text}"
    print(f"slice pp {sl.page_start+1}-{sl.page_end+1}, {len(text):,} chars\n")

    results = {}
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        print(f"--- {backend} ---", flush=True)
        t0 = time.perf_counter()
        try:
            if backend == "claude-cli":
                parsed, meta = run_claude_cli(prompt, args.timeout)
            else:
                parsed, meta = run_ollama(backend, prompt, args.timeout)
        except Exception as exc:
            print(f"  FAILED: {str(exc)[:200]}\n")
            results[backend] = {"error": str(exc)[:300]}
            continue
        wall = time.perf_counter() - t0
        ev = evaluate(parsed, text)
        cpv = meta["cost_usd"] / ev["nodes_verified"] if ev["nodes_verified"] else None
        results[backend] = {
            **ev, **{k: (round(v, 5) if isinstance(v, float) else v)
                     for k, v in meta.items()},
            "wall_s": round(wall, 1),
            "cost_per_verified_node_usd": round(cpv, 5) if cpv else None,
        }
        r = results[backend]
        print(f"  verificati {r['nodes_verified']}/{r['nodes_returned']} "
              f"({r['verified_rate_pct']}%), inventati {r['fabricated']}, "
              f"senza citazione {r['no_evidence']}")
        print(f"  archi {r['edges_kept']} ({r['relation_types']} tipi), "
              f"label_en {r['label_en_pct']}%")
        print(f"  ${r['cost_usd']} in {r['wall_s']}s -> "
              f"${r['cost_per_verified_node_usd']}/nodo verificato\n")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"slice": f"{sl.page_start+1}-{sl.page_end+1}", "chars": len(text),
             "results": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
