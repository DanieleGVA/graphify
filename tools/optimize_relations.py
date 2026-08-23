#!/usr/bin/env python3
"""Optimize link discovery: why does extraction collapse to one relation type?

Measured baseline: on the same slice Claude emits 5 relation types while
DeepSeek emits 1 ("references") — and the larger, pricier cloud models do not
close the gap, so this is not a capability that scales with model size.

Two hypotheses, tested independently:

  H1 (prompt)      the relation list is given without definitions or any demand
                   to differentiate, so models fall back to the generic label.
  H2 (vocabulary)  the vocabulary is inherited from graphify's code-analysis
                   origins (calls, implements, shares_data_with). On a prose
                   corpus most of it is inapplicable, leaving only "references"
                   — no prompt can fix a missing word.

Variant C stays domain-agnostic: `variant_of`, `requires`, `precedes`,
`part_of` describe any document corpus, not food specifically (CLAUDE.md
DOMAIN-AGNOSTIC rule).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

from graphify_ent.compact_schema import COMPACT_EXTRACTION_SYSTEM, parse_compact
from graphify_ent.file_slice import read_slice_text, slice_pdf

BASE_RELATIONS = ("calls", "implements", "references", "cites",
                  "conceptually_related_to", "shares_data_with",
                  "semantically_similar_to")

# --- A: current prompt (control) -------------------------------------------
PROMPT_A = COMPACT_EXTRACTION_SYSTEM

# --- B: same vocabulary, but defined, with an explicit demand to differentiate
PROMPT_B = COMPACT_EXTRACTION_SYSTEM.replace(
    """Rules: ids are lowercase [a-z0-9_]; relations are one of calls, implements,
references, cites, conceptually_related_to, shares_data_with,
semantically_similar_to; state the source file once in "f", never per row;""",
    """Relation types — choose the MOST SPECIFIC one that applies. Do not label
every edge "references": that is the fallback of last resort, and an extraction
where most edges are "references" is a failed extraction.
  cites                    A explicitly names or quotes B as a source
  conceptually_related_to  A and B belong to the same idea or family
  shares_data_with         A and B share a component, ingredient or input
  semantically_similar_to  A and B are near-equivalents or variants
  implements               A is a concrete realisation of the general B
  calls                    A invokes or depends on executing B
  references               generic mention, when none of the above fits

Rules: ids are lowercase [a-z0-9_]; state the source file once in "f", never per row;""")

# --- C: B plus domain-neutral prose relations the code vocabulary lacks ------
PROMPT_C = COMPACT_EXTRACTION_SYSTEM.replace(
    """Rules: ids are lowercase [a-z0-9_]; relations are one of calls, implements,
references, cites, conceptually_related_to, shares_data_with,
semantically_similar_to; state the source file once in "f", never per row;""",
    """Relation types — choose the MOST SPECIFIC one that applies. Do not label
every edge "references": that is the fallback of last resort, and an extraction
where most edges are "references" is a failed extraction.
  variant_of               A is a variation or derivative of B
  requires                 A needs B as a component, input or precondition
  precedes                 A must happen before B in a sequence
  part_of                  A is a constituent of the larger B
  cites                    A explicitly names or quotes B as a source
  conceptually_related_to  A and B belong to the same idea or family
  shares_data_with         A and B share a component or input
  semantically_similar_to  A and B are near-equivalents
  implements               A is a concrete realisation of the general B
  references               generic mention, when none of the above fits

Rules: ids are lowercase [a-z0-9_]; state the source file once in "f", never per row;""")

# --- D: C plus an explicit anti-monopoly constraint --------------------------
# Measured: with A and B the model picks ONE label and applies it to every edge
# (A -> all "calls", B -> all "shares_data_with"). Definitions alone do not
# differentiate; the constraint has to be quantitative and checkable.
PROMPT_D = PROMPT_C.replace(
    """Rules: ids are lowercase [a-z0-9_]; state the source file once in "f", never per row;""",
    """CRITICAL — relation diversity is checked: if more than half of your edges
carry the same relation type, the extraction is rejected and must be redone.
Before emitting each edge, ask what the text actually asserts between the two
entities, and pick the type from that. A list of ingredients is `requires`; a
section heading over its contents is `part_of`; a named variation of a base
preparation is `variant_of`; a step that must follow another is `precedes`.
Different edges in the same response MUST carry different types.

Rules: ids are lowercase [a-z0-9_]; state the source file once in "f", never per row;""")

PROMPTS = {"A_control": PROMPT_A, "B_defined": PROMPT_B, "C_extended": PROMPT_C,
           "D_diversity": PROMPT_D}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def run(model: str, system: str, prompt: str, timeout: int = 900) -> tuple[str, dict]:
    payload = json.dumps({
        "model": model, "stream": False, "think": False,
        "options": {"temperature": 0, "num_ctx": 32768, "num_predict": 16000},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        env = json.loads(resp.read())
    return (env.get("message") or {}).get("content") or "", {
        "out_tokens": env.get("eval_count", 0),
        "truncated": env.get("done_reason") == "length",
    }


def score(content: str, source: str) -> dict:
    nodes, edges = parse_compact(content)
    src = norm(source)
    kept = [n for n in nodes if norm(n.get("evidence", "")) and norm(n["evidence"]) in src]
    ids = {n["id"] for n in kept}
    ke = [e for e in edges if e["source"] in ids and e["target"] in ids]
    rels = Counter(e["relation"] for e in ke)
    generic = rels.get("references", 0)
    return {
        "nodes_verified": len(kept), "nodes_returned": len(nodes),
        "verified_pct": round(100 * len(kept) / len(nodes), 1) if nodes else 0.0,
        "edges": len(ke), "relation_types": len(rels),
        "generic_share_pct": round(100 * generic / len(ke), 1) if ke else 0.0,
        "distribution": dict(rels.most_common()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--slice-index", type=int, default=20)
    ap.add_argument("--models", default="deepseek-v4-flash:cloud")
    ap.add_argument("--variants", default="A_control,B_defined,C_extended")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    sl = slice_pdf(args.pdf, 20_000)[args.slice_index]
    text = read_slice_text(sl)
    user = f"=== {args.pdf.name} (pages {sl.page_start+1}-{sl.page_end+1}) ===\n{text}"

    out: dict = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        out[model] = {}
        for vname in [v.strip() for v in args.variants.split(",") if v.strip()]:
            t0 = time.perf_counter()
            try:
                content, meta = run(model, PROMPTS[vname], user)
            except Exception as exc:
                print(f"  {model} / {vname}: FALLITO {str(exc)[:120]}")
                out[model][vname] = {"error": str(exc)[:200]}
                continue
            s = score(content, text)
            s.update(meta)
            s["wall_s"] = round(time.perf_counter() - t0, 1)
            out[model][vname] = s
            print(f"  {model:<26} {vname:<12} "
                  f"nodi {s['nodes_verified']:>3}/{s['nodes_returned']:<3} "
                  f"({s['verified_pct']:>5}%)  archi {s['edges']:>3}  "
                  f"tipi {s['relation_types']}  generico {s['generic_share_pct']:>5}%  "
                  f"{s['wall_s']:>5}s", flush=True)
            print(f"      {s['distribution']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
