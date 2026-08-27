#!/usr/bin/env python3
"""Decide which books may enter the graph, and prove the decision.

A book whose text is quietly corrupt is worse than a missing book: it fills the
graph with facts that read as authoritative and are wrong. "black sticky rrce
flour or whrte strcky nee flour" is a real line from a real cookbook PDF whose
quantities are perfectly legible and whose ingredients are not.

Pattern heuristics miss that: `rrce` and `nee` are short and consonant-light,
so a consonant-cluster rule scores the page at 98% clean. What catches it is
reading, so the judge here is a model — cheap, multilingual (the corpus is
EN/FR/IT), and asked one narrow question per sample: is this text intact enough
that a fact extracted from it would be trustworthy?

Two independent signals, both reported:
  * `chars_per_page` — a scan with no text layer at all scores zero and needs
    OCR before it can be judged on content.
  * `legibility` — the model's median score over sampled prose pages.

Nothing is admitted on one signal alone, and every score is written out per
page so a human can overrule the machine.

    python tools/triage_corpus.py --corpus <dir> --json ../evidence/T90/triage.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

SYSTEM = (
    "Giudichi la QUALITÀ DI ESTRAZIONE di testo da libri (PDF/EPUB), non il contenuto.\n"
    "Il testo può essere in inglese, francese o italiano.\n"
    "Rispondi SOLO con un numero da 0 a 100:\n"
    "100 = testo perfettamente leggibile, parole intatte\n"
    " 70 = qualche refuso o carattere strano, ma tutte le parole sono riconoscibili\n"
    " 40 = molte parole corrotte dall'OCR (es. 'rrce' per 'rice', 'strcky' per 'sticky'),\n"
    "      un fatto estratto da qui sarebbe inaffidabile\n"
    "  0 = illeggibile, caratteri cifrati o casuali\n"
    "Guarda soprattutto i NOMI (ingredienti, tecniche): se sono corrotti il punteggio "
    "è basso anche se i numeri si leggono. Rispondi col solo numero."
)


def judge(model: str, text: str, timeout: int = 180) -> int | None:
    payload = json.dumps({
        "model": model, "stream": False, "think": False,
        "options": {"temperature": 0, "num_predict": 8},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": text[:2500]}],
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            env = json.loads(r.read())
        m = re.search(r"\d{1,3}", (env.get("message") or {}).get("content", ""))
        return min(100, int(m.group())) if m else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-v4-flash:cloud")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--min-legibility", type=int, default=85)
    ap.add_argument("--min-prose-pages", type=int, default=30,
                    help="sotto questo numero di pagine con prosa non è un libro di testo")
    ap.add_argument("--json", type=Path, default=Path("../evidence/T90/triage.json"))
    args = ap.parse_args()

    import fitz

    files = sorted([p for p in args.corpus.iterdir()
                    if p.suffix.lower() in (".pdf", ".epub")])
    out = []
    print(f"{'libro':<50} {'pag':>5} {'prosa':>6} {'car/pag':>8} {'legg.':>6}  esito")
    print("-" * 92)
    for f in files:
        doc = fitz.open(f)
        n = doc.page_count
        # Sample across the body, skipping front and back matter.
        idx = [int(n * 0.08) + i * max(1, int(n * 0.84 / 40)) for i in range(40)]
        prose, chars = [], 0
        for i in [i for i in idx if i < n]:
            t = doc[i].get_text()
            chars += len(t)
            if len(t.strip()) > 400:
                prose.append(t)
        # count prose pages over the whole book, not just the sample
        prose_total = sum(1 for i in range(0, n, max(1, n // 120))
                          if len(doc[i].get_text().strip()) > 400)
        prose_total = int(prose_total * n / max(1, len(range(0, n, max(1, n // 120)))))
        doc.close()

        scores = []
        t0 = time.perf_counter()
        for t in prose[: args.samples]:
            s = judge(args.model, re.sub(r"\s+", " ", t))
            if s is not None:
                scores.append(s)
        legibility = statistics.median(scores) if scores else 0
        sampled = len([i for i in idx if i < n])
        cpp = chars / max(sampled, 1)
        # Density is measured over the pages that HAVE prose, not over every
        # page: an art-directed cookbook can be a third photographs, and
        # dividing its text by the plate pages made a perfectly legible book
        # (100/100, 1,443 chars per text page) look like an unOCR'd scan.
        cpp_prose = (chars / len(prose)) if prose else 0

        no_text = not prose or cpp_prose < 250
        admitted = (legibility >= args.min_legibility
                    and prose_total >= args.min_prose_pages
                    and not no_text)
        reason = ("ok" if admitted else
                  "nessun testo (serve OCR)" if no_text else
                  f"testo corrotto (leggibilità {legibility})" if legibility < args.min_legibility
                  else "troppo poca prosa")
        out.append({"file": f.name, "format": f.suffix.lower().lstrip("."),
                    "pages": n, "prose_pages_est": prose_total,
                    "chars_per_page": round(cpp),
                    "chars_per_prose_page": round(cpp_prose),
                    "legibility_median": legibility,
                    "legibility_scores": scores, "admitted": admitted, "reason": reason,
                    "judge_seconds": round(time.perf_counter() - t0, 1)})
        print(f"{f.name[:48]:<50} {n:>5} {prose_total:>6} {cpp:>8.0f} {legibility:>6}  "
              f"{'AMMESSO' if admitted else 'ESCLUSO — ' + reason}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(
        {"corpus": str(args.corpus), "judge": args.model,
         "gate": {"min_legibility": args.min_legibility,
                  "min_prose_pages": args.min_prose_pages,
                  "min_chars_per_prose_page": 250},
         "books": out}, indent=1, ensure_ascii=False))
    ok = sum(1 for b in out if b["admitted"])
    print(f"\nammessi {ok}/{len(out)} -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
