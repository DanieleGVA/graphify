#!/usr/bin/env python3
"""Layer 3 — end-to-end: build the graph upstream graphify would build, then
answer the same golden questions through the same retriever.

Arm A1 (this script) runs upstream's pipeline steel-manned:
  * upstream's own PDF reader (`detect.extract_pdf_text`, pypdf),
  * upstream's 20k per-file cap (`llm._read_files`),
  * upstream's verbose JSON schema, loaded verbatim from the pre-fork commit,
  * upstream's code-analysis relation vocabulary.

Two deliberate helps, both stated in the output so they cannot be mistaken for
upstream capabilities:

  1. Truncated responses are salvaged, exactly as ENTERPRIPHY salvages its own.
  2. Upstream's schema has no `evidence` field, so retrieval's fail-closed rule
     would block every node it produces and the arm would score zero on a
     policy technicality rather than on content. Evidence is therefore
     RECONSTRUCTED for it: the source sentence containing the node label. This
     is generous — upstream cannot do it — and the count is reported.

What is NOT given to it: `label_en` / `lang`. Cross-language answering comes
from ENTERPRIPHY's canonical-label layer; upstream has no such field, and
inventing one would measure this script, not graphify.

Writes graph.json + loads into the TEST Neo4j instance, never the pilot graph.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from bench_vs_upstream import CHAR_CAP, PRICING, parse_upstream, pypdf_text, upstream_prompt

_SENT = re.compile(r"[^.!?\n]{20,400}[.!?]")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def reconstruct_evidence(label: str, source: str, sentences: list[str]) -> str:
    """Best sentence from the source that literally contains the label."""
    lab = norm(label)
    if not lab or lab not in norm(source):
        return ""
    for s in sentences:
        if lab in norm(s):
            return s.strip()[:1000]
    return ""


def run(model: str, system: str, user: str, timeout: int = 900) -> tuple[str, dict]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash:cloud")
    ap.add_argument("--corpus", type=Path, default=Path("../pilot"))
    ap.add_argument("--out", type=Path, default=Path("../evidence/T70/upstream-graph.json"))
    ap.add_argument("--json", type=Path,
                    default=Path("../evidence/T70/bench-vs-upstream-e2e.json"))
    args = ap.parse_args()

    sys_prompt = upstream_prompt()
    pdfs = sorted(args.corpus.glob("*.pdf"))
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    stats = {"model": args.model, "cap": CHAR_CAP, "books": [], "helps": {}}
    reconstructed = 0
    seen: set[str] = set()

    print("=" * 92)
    print("LIVELLO 3 / arm A1 — graphify rinforzato: estrazione")
    print("=" * 92)
    for pdf in pdfs:
        text = pypdf_text(pdf)[:CHAR_CAP]            # upstream's reader + upstream's cap
        user = f"=== {pdf.name} ===\n{text}"          # upstream's _read_files framing
        t0 = time.perf_counter()
        try:
            content, meta = run(args.model, sys_prompt, user)
        except Exception as exc:
            print(f"  {pdf.name[:34]:<36} FALLITO {str(exc)[:60]}")
            continue
        nodes, edges = parse_upstream(content)
        sentences = _SENT.findall(text)
        kept = []
        for n in nodes:
            nid = str(n.get("id") or "").strip()
            if not nid or nid in seen:
                continue
            ev = reconstruct_evidence(n.get("label", ""), text, sentences)
            if ev:
                reconstructed += 1
            seen.add(nid)
            kept.append({
                "id": nid, "label": n.get("label") or nid,
                "label_orig": n.get("label") or nid,
                "file_type": "document", "source_file": pdf.name,
                "evidence": ev, "text_excerpt": ev[:1000],
                "confidence": "EXTRACTED", "extraction_method": "upstream_llm",
            })
        all_nodes.extend(kept)
        ids = {n["id"] for n in kept}
        ke = [{"source": e.get("source"), "target": e.get("target"),
               "relation": e.get("relation") or "references",
               "confidence": e.get("confidence") or "INFERRED",
               "confidence_score": e.get("confidence_score") or 0.7,
               "source_file": pdf.name, "weight": 1.0}
              for e in edges if e.get("source") in ids and e.get("target") in ids]
        all_edges.extend(ke)
        stats["books"].append({
            "book": pdf.name[:34], "chars_sent": len(text), "nodes": len(kept),
            "edges": len(ke), "truncated": meta["truncated"],
            "out_tokens": meta["out_tokens"], "wall_s": round(time.perf_counter() - t0, 1),
        })
        print(f"  {pdf.name[:34]:<36} inviati {len(text):>6,} char  "
              f"nodi {len(kept):>4}  archi {len(ke):>4}  "
              f"troncata {'si' if meta['truncated'] else 'no':<3} "
              f"{round(time.perf_counter()-t0,1):>5}s", flush=True)

    p = PRICING.get(args.model.split(":")[0])
    tot_in = sum(b.get("out_tokens", 0) for b in stats["books"])
    stats["totals"] = {"nodes": len(all_nodes), "edges": len(all_edges),
                       "chars_sent": sum(b["chars_sent"] for b in stats["books"]),
                       "truncated_books": sum(1 for b in stats["books"] if b["truncated"])}
    stats["helps"] = {
        "salvage_applied": True,
        "evidence_reconstructed_nodes": reconstructed,
        "evidence_reconstructed_pct": round(100 * reconstructed / max(len(all_nodes), 1), 1),
        "note": "upstream's schema has no evidence field; without this help the "
                "fail-closed retrieval rule would block every node it produced.",
    }
    print(f"\ntotale: {len(all_nodes)} nodi, {len(all_edges)} archi, "
          f"{stats['totals']['chars_sent']:,} caratteri inviati")
    print(f"aiuto concesso: prove ricostruite per {reconstructed}/{len(all_nodes)} nodi "
          f"({stats['helps']['evidence_reconstructed_pct']}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"nodes": all_nodes, "edges": all_edges},
                                   ensure_ascii=False))
    args.json.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"grafo -> {args.out}\nstatistiche -> {args.json}")

    if not os.environ.get("NEO4J_URI", "").endswith("7689"):
        print("\nATTENZIONE: NEO4J_URI non punta all'istanza di test (porta 7689). "
              "Non carico, per non toccare il grafo pilota.")
        return 0

    from graphify_ent.loader import Neo4jLoader
    loader = Neo4jLoader()
    print(f"\ncarico in {loader.uri} / {loader.database} (istanza di test)")
    loader.wipe()
    # The eval harness filters on domain="pilot"; the baseline must carry the
    # same tag or it is invisible to retrieval and scores zero on a harness
    # artefact rather than on content.
    ls = loader.load(args.out, domain="pilot")
    print(json.dumps(ls.as_dict(), indent=2))
    loader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
