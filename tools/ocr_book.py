#!/usr/bin/env python3
"""Transcribe a scanned book with a vision model, page by page, into Markdown.

For books whose text layer is missing or destroyed. Measured on the Ducasse
*Desserts et Pâtisseries* scan (586 pages, zero extractable characters):
tesseract returned nothing at all on pages where the vision model read the
headings, and on the pages where both produced text the model kept the accents
and the section title tesseract dropped.

Design follows what the corpus load taught:

  * **Resumable.** 586 pages is long enough that a laptop lid must not cost the
    run. Every finished page is appended to a JSONL checkpoint and skipped on
    restart.
  * **Rate-limit aware.** The endpoint pushes back; the concurrency window
    narrows on 429 and widens after a clean run (`AdaptiveLimiter`).
  * **A blank page is a result, not a failure.** Plates and dividers carry no
    text and the model is told to say so, so an empty transcription is
    distinguishable from a page that failed.

The output is Markdown with a `<!-- page N -->` marker before each page, so any
later step can still cite a page number — which is the whole point of keeping
provenance.

    python tools/ocr_book.py "<file.pdf>" --out ../evidence/T91/ducasse.md
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from graphify_ent.ratelimit import AdaptiveLimiter

PROMPT = (
    "Trascrivi FEDELMENTE tutto il testo di questa pagina di libro di cucina. "
    "Rispetta la struttura: titoli, elenchi di ingredienti con le quantità e le unità, "
    "passaggi del procedimento. Usa markdown: ## per i titoli, - per gli elenchi. "
    "Conserva la lingua originale e gli accenti. "
    "NON tradurre, NON riassumere, NON aggiungere commenti tuoi. "
    "Se la pagina non contiene testo (fotografia, pagina bianca), "
    "rispondi esattamente: [PAGINA SENZA TESTO]"
)
BLANK = "[PAGINA SENZA TESTO]"
_print_lock = threading.Lock()
_ckpt_lock = threading.Lock()


def transcribe(model: str, png: bytes, timeout: int,
               limiter: AdaptiveLimiter, attempts: int = 5) -> tuple[str, dict]:
    last = None
    for a in range(attempts):
        try:
            payload = json.dumps({
                "model": model, "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": 4000},
                "messages": [{"role": "user", "content": PROMPT,
                              "images": [base64.b64encode(png).decode()]}],
            }).encode()
            req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                         headers={"Content-Type": "application/json"})
            with limiter.slot():
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    env = json.loads(r.read())
            limiter.succeeded()
            return ((env.get("message") or {}).get("content") or "").strip(), {
                "out_tokens": env.get("eval_count", 0),
                "truncated": env.get("done_reason") == "length",
            }
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                ra = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    ra = float(ra) if ra else None
                except ValueError:
                    ra = None
                time.sleep(limiter.rejected(ra, a))
            else:
                time.sleep(2 ** a * 2)
        except Exception as exc:
            last = exc
            time.sleep(2 ** a * 2)
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:120]}")


def render(pdf_path: Path, index: int, dpi: int) -> bytes:
    import fitz
    with fitz.open(pdf_path) as doc:
        return doc[index].get_pixmap(dpi=dpi).tobytes("png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--model", default="qwen3.5:cloud")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="prova: solo le prime N pagine")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import fitz
    with fitz.open(args.pdf) as doc:
        total = doc.page_count
    pages = list(range(total))[: args.limit or total]
    ckpt = args.checkpoint or args.out.with_suffix(".jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    done: dict[int, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["page"]] = rec
    todo = [p for p in pages if p not in done]
    print(f"{args.pdf.name}\npagine {len(pages)}  già fatte {len(done)}  da fare {len(todo)}",
          flush=True)

    limiter = AdaptiveLimiter(start=args.workers, minimum=2,
                              maximum=max(args.workers, args.max_workers))
    fh = ckpt.open("a")
    t0 = time.perf_counter()
    failures: list[int] = []
    completed = 0

    def work(i: int) -> dict:
        png = render(args.pdf, i, args.dpi)
        t = time.perf_counter()
        text, meta = transcribe(args.model, png, args.timeout, limiter)
        return {"page": i + 1, "index": i, "text": text,
                "blank": BLANK in text or not text,
                "chars": len(text), "seconds": round(time.perf_counter() - t, 1), **meta}

    with ThreadPoolExecutor(max_workers=max(args.workers, args.max_workers)) as pool:
        futs = {pool.submit(work, i): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                failures.append(i + 1)
                with _print_lock:
                    print(f"  FALLITA p.{i+1}: {str(exc)[:80]}", flush=True)
                continue
            with _ckpt_lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[rec["page"]] = rec
            completed += 1
            if completed % 20 == 0 or completed == len(todo):
                el = time.perf_counter() - t0
                rate = completed / el * 60
                with _print_lock:
                    print(f"  {completed}/{len(todo)}  {rate:.1f} pag/min  "
                          f"restano ~{(len(todo)-completed)/max(rate,0.1):.0f} min  "
                          f"[conc {limiter.limit}, 429 {limiter.rejections}]", flush=True)
    fh.close()

    # --- assemble ---------------------------------------------------------
    body = [f"# {args.pdf.stem}", "",
            f"> Trascrizione OCR di una scansione senza layer di testo. "
            f"Modello: `{args.model}`, {args.dpi} dpi. "
            f"Ogni pagina conserva il proprio numero, così una citazione resta verificabile.",
            ""]
    written = blank = 0
    for p in sorted(done):
        rec = done[p]
        body.append(f"<!-- page {p} -->")
        if rec["blank"]:
            blank += 1
            body.append(f"*[pagina {p}: nessun testo]*\n")
        else:
            written += 1
            body.append(rec["text"] + "\n")
    args.out.write_text("\n".join(body), encoding="utf-8")

    chars = sum(r["chars"] for r in done.values() if not r["blank"])
    stats = {"pdf": str(args.pdf), "model": args.model, "dpi": args.dpi,
             "pages_total": len(pages), "pages_done": len(done),
             "pages_failed": len(failures), "failed_pages": failures[:40],
             "pages_with_text": written, "pages_blank": blank,
             "chars": chars, "chars_per_text_page": round(chars / max(written, 1)),
             "out_tokens": sum(r.get("out_tokens", 0) for r in done.values()),
             "truncated": sum(1 for r in done.values() if r.get("truncated")),
             "minutes": round((time.perf_counter() - t0) / 60, 1),
             "rate_limit": limiter.stats()}
    args.out.with_suffix(".stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nmarkdown -> {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
