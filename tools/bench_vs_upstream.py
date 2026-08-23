#!/usr/bin/env python3
"""Benchmark ENTERPRIPHY against upstream graphify on the same PDF corpus.

Three arms, because a two-arm comparison here would be a strawman:

  A0  graphify as-is.       `_read_files` reads every file with
                            `read_text(encoding="utf-8", errors="replace")`.
                            On a PDF that is the binary container, not the book:
                            `extract_pdf_text` (pypdf) exists in detect.py but is
                            wired only to word counting, never to extraction.
                            Reported once to establish the true starting point.

  A1  graphify steel-manned. Upstream plus the minimal, obvious fix it would need
                            to accept PDFs at all: route them through its own
                            `extract_pdf_text`. Keeps upstream's 20k per-file cap,
                            upstream's verbose JSON schema and upstream's
                            code-analysis relation vocabulary. This is the honest
                            baseline — it isolates what ENTERPRIPHY contributes
                            beyond five minutes of obvious work.

  B   ENTERPRIPHY.          Page-based slicing (full coverage), compact schema,
                            prose relation vocabulary, mandatory evidence binding.

The upstream prompt is loaded from git at the pre-fork commit rather than copied
here, so the baseline cannot silently drift as this repo edits graphify/.

Layer 1 (coverage) is deterministic and needs no model. Layer 2 feeds A1 and B
the *identical* slice text, which neutralises coverage and isolates the effect
of schema, vocabulary and evidence binding.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

from graphify_ent.compact_schema import COMPACT_EXTRACTION_SYSTEM, parse_compact
from graphify_ent.file_slice import read_slice_text, slice_pdf

#: Commit predating this repo's graphify/ patches. The upstream prompt is read
#: from here so the baseline is provably pristine.
UPSTREAM_REF = "f755aca"
CHAR_CAP = 20_000
PRICING = {"deepseek-v4-flash": {"input": 0.14, "output": 0.28}}

BOOK_SLICES = [
    ("Escoffier  Le Guide Culinaire - Georges Auguste Escoffier.pdf", "fr", [20, 60]),
    ("The Professional Chef - The Culinary Institute of America.pdf", "en", [20, 50]),
    ("China The Cookbook - Kei Lum Chan, Diora Fong Chan.pdf", "en", [15, 45]),
]


def upstream_prompt(ref: str = UPSTREAM_REF) -> str:
    """Load upstream's `_EXTRACTION_SYSTEM` verbatim from git."""
    src = subprocess.run(["git", "show", f"{ref}:graphify/llm.py"],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r'_EXTRACTION_SYSTEM = """\\\n(.*?)"""', src, re.S)
    if not m:
        raise SystemExit(f"non trovo _EXTRACTION_SYSTEM in {ref}:graphify/llm.py")
    return m.group(1)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


# --------------------------------------------------------------------------
# Layer 1 — coverage (deterministic)
# --------------------------------------------------------------------------

def pypdf_text(path: Path) -> str:
    """Upstream's own PDF reader (detect.extract_pdf_text), used verbatim."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(t for p in reader.pages if (t := p.extract_text()))
    except Exception:
        return ""


def coverage(pdfs: list[Path]) -> list[dict]:
    rows = []
    for pdf in pdfs:
        slices = slice_pdf(pdf, CHAR_CAP)
        ours = sum(len(read_slice_text(s)) for s in slices)
        raw = pdf.read_text(encoding="utf-8", errors="replace")[:CHAR_CAP]
        # A0 sees the PDF container. Count how much of it is even legible.
        a0_legible = len(raw) - raw.count("�")
        up = pypdf_text(pdf)
        rows.append({
            "book": pdf.name[:34], "pages": len(slices) and slices[-1].page_end + 1,
            "chars_total_ours": ours, "chars_total_pypdf": len(up),
            "a0_chars_sent": len(raw), "a0_replacement_chars": raw.count("�"),
            "a0_legible_chars": a0_legible,
            "a1_chars_sent": min(CHAR_CAP, len(up)),
            "b_chars_sent": ours, "slices": len(slices),
            "a1_coverage_pct": round(100 * min(CHAR_CAP, len(up)) / ours, 2) if ours else 0.0,
        })
    return rows


# --------------------------------------------------------------------------
# Layer 2 — extraction quality on identical input
# --------------------------------------------------------------------------

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


def _salvage_upstream(text: str) -> dict:
    """Recover complete objects from an upstream response truncated mid-JSON.

    ENTERPRIPHY's parser salvages truncated rows (`compact_schema._salvage`).
    Giving that to our arm and not to upstream's would manufacture the very gap
    the benchmark is meant to measure, so upstream gets the same treatment:
    walk each array, keep every object that closed.
    """
    out: dict = {"nodes": [], "edges": []}
    for key in ("nodes", "edges"):
        section = re.search(rf'"{key}"\s*:\s*\[', text)
        if not section:
            continue
        i, depth, start = section.end(), 0, None
        while i < len(text):
            ch = text[i]
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        out[key].append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
            elif ch == "]" and depth == 0:
                break
            i += 1
    return out


def parse_upstream(raw: str) -> tuple[list[dict], list[dict]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text[4:] if text.startswith("json") else text
        text = text.rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _salvage_upstream(text)
    return data.get("nodes") or [], data.get("edges") or []


def score(nodes: list[dict], edges: list[dict], source: str, arm: str) -> dict:
    """Common metrics for both arms.

    `label_grounded` is the only hallucination check upstream's schema permits —
    it has no evidence field — so it is computed for BOTH arms to keep the
    comparison fair. `evidence_verified` is the stricter check ENTERPRIPHY's
    schema enables, and is structurally unavailable to upstream (reported as
    None, never 0.0, so it cannot be read as a measured failure).
    """
    src = norm(source)
    grounded = [n for n in nodes if norm(n.get("label", "")) and norm(n["label"]) in src]
    if arm == "B":
        verified = [n for n in nodes
                    if norm(n.get("evidence", "")) and norm(n["evidence"]) in src]
        ev_pct = round(100 * len(verified) / len(nodes), 1) if nodes else 0.0
        keep_ids = {n["id"] for n in verified}
    else:
        ev_pct = None
        keep_ids = {n.get("id") for n in nodes}
    ke = [e for e in edges
          if e.get("source") in keep_ids and e.get("target") in keep_ids]
    rels = Counter(e.get("relation", "references") for e in ke)
    return {
        "nodes_returned": len(nodes),
        "label_grounded": len(grounded),
        "label_grounded_pct": round(100 * len(grounded) / len(nodes), 1) if nodes else 0.0,
        "evidence_verified_pct": ev_pct,
        "edges": len(ke),
        "relation_types": len(rels),
        "generic_pct": round(100 * rels.get("references", 0) / len(ke), 1) if ke else 0.0,
        "dominant_pct": round(100 * rels.most_common(1)[0][1] / len(ke), 1) if ke else 0.0,
        "distribution": dict(rels.most_common(5)),
    }


def cost_usd(model: str, meta: dict) -> float | None:
    p = PRICING.get(model.split(":")[0])
    if not p:
        return None
    return round(meta["in_tokens"] / 1e6 * p["input"]
                 + meta["out_tokens"] / 1e6 * p["output"], 6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash:cloud")
    ap.add_argument("--corpus", type=Path, default=Path("../pilot"))
    ap.add_argument("--layer", default="1,2")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    layers = {s.strip() for s in args.layer.split(",")}
    out: dict = {"upstream_ref": UPSTREAM_REF, "model": args.model}

    if "1" in layers:
        pdfs = sorted(args.corpus.glob("*.pdf"))
        rows = coverage(pdfs)
        out["coverage"] = rows
        print("=" * 92)
        print("LIVELLO 1 — copertura del testo (deterministico, nessun modello)")
        print("=" * 92)
        print(f"{'libro':<36}{'pagine':>7}{'char totali':>13}{'A1 inviati':>12}"
              f"{'A1 %':>8}{'B inviati':>12}{'B %':>7}")
        for r in rows:
            print(f"{r['book']:<36}{r['pages']:>7}{r['chars_total_ours']:>13,}"
                  f"{r['a1_chars_sent']:>12,}{r['a1_coverage_pct']:>7.2f}%"
                  f"{r['b_chars_sent']:>12,}{100.0:>6.1f}%")
        tot = sum(r["chars_total_ours"] for r in rows)
        a1 = sum(r["a1_chars_sent"] for r in rows)
        a0_leg = sum(r["a0_legible_chars"] for r in rows)
        a0_repl = sum(r["a0_replacement_chars"] for r in rows)
        print("-" * 92)
        print(f"{'TOTALE':<36}{sum(r['pages'] for r in rows):>7}{tot:>13,}"
              f"{a1:>12,}{100*a1/tot:>7.2f}%{tot:>12,}{100.0:>6.1f}%")
        print(f"\nA0 (graphify così com'è): invia {sum(r['a0_chars_sent'] for r in rows):,} "
              f"caratteri di contenitore PDF, di cui {a0_repl:,} illeggibili "
              f"({100*a0_repl/max(a0_leg+a0_repl,1):.1f}%) e zero testo dei libri.")
        pypdf_tot = sum(r["chars_total_pypdf"] for r in rows)
        print(f"Resa dell'estrattore: pypdf (upstream) {pypdf_tot:,} caratteri contro "
              f"PyMuPDF (nostro) {tot:,} — {100*pypdf_tot/tot:.1f}%.")
        out["coverage_totals"] = {"chars_total": tot, "a1_sent": a1,
                                  "a1_pct": round(100 * a1 / tot, 3),
                                  "pypdf_total": pypdf_tot}

    if "2" in layers:
        up_sys = upstream_prompt()
        print("\n" + "=" * 92)
        print("LIVELLO 2 — qualità dell'estrazione a parità di testo in ingresso")
        print(f"prompt upstream caricato da {UPSTREAM_REF} ({len(up_sys)} char); "
              f"nostro {len(COMPACT_EXTRACTION_SYSTEM)} char")
        print("=" * 92)
        rows = []
        for fname, lang, idxs in BOOK_SLICES:
            pdf = args.corpus / fname
            if not pdf.exists():
                print(f"manca {fname}")
                continue
            slices = slice_pdf(pdf, CHAR_CAP)
            for si in idxs:
                if si >= len(slices):
                    continue
                sl = slices[si]
                text = read_slice_text(sl)
                user = (f"=== {pdf.name} (pages {sl.page_start+1}-{sl.page_end+1}) ===\n"
                        f"{text}")
                for arm, sysprompt in (("A1_graphify", up_sys), ("B_enterpriphy",
                                                                 COMPACT_EXTRACTION_SYSTEM)):
                    t0 = time.perf_counter()
                    try:
                        content, meta = run(args.model, sysprompt, user)
                    except Exception as exc:
                        print(f"  {pdf.name[:20]} s{si} {arm}: FALLITO {str(exc)[:70]}")
                        continue
                    if arm.startswith("A1"):
                        n, e = parse_upstream(content)
                    else:
                        n, e = parse_compact(content)
                    s = score(n, e, text, "A1" if arm.startswith("A1") else "B")
                    s.update(meta)
                    s.update({"book": pdf.name[:24], "lang": lang, "slice": si,
                              "arm": arm, "wall_s": round(time.perf_counter() - t0, 1),
                              "cost_usd": cost_usd(args.model, meta)})
                    rows.append(s)
                    ev = ("n/d" if s["evidence_verified_pct"] is None
                          else f"{s['evidence_verified_pct']:.1f}%")
                    print(f"  {pdf.name[:22]:<24} s{si:<3} {arm:<14} "
                          f"nodi {s['nodes_returned']:>4}  ancorati "
                          f"{s['label_grounded_pct']:>5.1f}%  prove {ev:>6}  "
                          f"archi {s['edges']:>4}  generico {s['generic_pct']:>5.1f}%  "
                          f"out-tok {s['out_tokens']:>6}  {s['wall_s']:>5}s", flush=True)
        out["layer2"] = rows

        print("\n" + "-" * 92)
        for arm in ("A1_graphify", "B_enterpriphy"):
            sel = [r for r in rows if r["arm"] == arm]
            if not sel:
                continue
            def m(k):
                return statistics.mean([r[k] for r in sel])
            costs = [r["cost_usd"] for r in sel if r["cost_usd"] is not None]
            nodes = sum(r["nodes_returned"] for r in sel)
            gnodes = sum(r["label_grounded"] for r in sel)
            trunc = sum(1 for r in sel if r["truncated"])
            print(f"{arm:<15} n={len(sel)}  nodi {nodes:>5}  ancorati {gnodes:>5} "
                  f"({100*gnodes/max(nodes,1):5.1f}%)  archi {sum(r['edges'] for r in sel):>5}  "
                  f"generico {m('generic_pct'):5.1f}%  "
                  f"out-tok/nodo {sum(r['out_tokens'] for r in sel)/max(nodes,1):6.1f}  "
                  f"costo {sum(costs):.4f}$  {m('wall_s'):5.1f}s/porzione  "
                  f"troncate {trunc}/{len(sel)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\ndati grezzi -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
