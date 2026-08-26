#!/usr/bin/env python3
"""Q2 — answer correctness (blueprint §3), the one gate figure never measured.

The pipeline under test is the whole one, not retrieval alone: the graph
returns evidence, a model answers **bound to that evidence**, and the answer is
graded against a hand-authored key. An answer the corpus does not support must
come back as a refusal — for the four unanswerable pairs that is the only
correct outcome, and any substantive answer is scored wrong.

Two graders, deliberately:

  * **mechanical** — every key group must appear in the answer (accent- and
    case-folded). Recomputable by ENTF-15 with no model and no network, which
    is the point: the headline figure must not rest on an LLM's opinion.
  * **judged** — an independent model (default qwen3.5, never the answerer)
    rates CORRETTA / INCOMPLETA / SBAGLIATA / RIFIUTO against the expected
    answer. Catches the case the key match cannot: right numbers, wrong claim.

Q2 is reported as the *lower* of the two. Maker/checker also applies to models:
the answerer is DeepSeek (the programme's extraction model), the judge is not.

    python tools/bench_q2.py --check      # verify the golden set against the PDFs
    python tools/bench_q2.py              # measure
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
import urllib.request
from pathlib import Path

ANSWER_SYSTEM = (
    "Sei un assistente che risponde ESCLUSIVAMENTE usando i passaggi forniti. "
    "Regole assolute:\n"
    "1. Se i passaggi contengono la risposta, rispondi in modo diretto e completo, "
    "riportando cifre e unità come le scrive la fonte.\n"
    "2. Se i passaggi NON contengono la risposta, rispondi esattamente: NON_SUPPORTATO\n"
    "3. Non usare conoscenza tua: qualunque cosa non sia nei passaggi non esiste.\n"
    "4. Rispondi nella lingua della domanda, in due frasi al massimo."
)

JUDGE_SYSTEM = (
    "Sei un giudice di valutazione. Ricevi una DOMANDA, la RISPOSTA ATTESA (verità "
    "stabilita leggendo la fonte) e la RISPOSTA DEL SISTEMA. Classifica con UNA parola:\n"
    "CORRETTA — dice il fatto atteso, completo e senza errori\n"
    "INCOMPLETA — corretta ma le manca una parte sostanziale dell'atteso\n"
    "SBAGLIATA — contraddice l'atteso o afferma altro\n"
    "RIFIUTO — il sistema dichiara di non poter rispondere\n"
    "Rispondi con la sola parola, senza spiegazioni."
)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[\s­]+", " ", s).strip().lower()


def matches(answer: str, keys: list[list[str]]) -> bool:
    body = fold(answer)
    return all(any(fold(alt) in body for alt in group) for group in keys)


def call(model: str, system: str, user: str, timeout: int = 300) -> str:
    payload = json.dumps({
        "model": model, "stream": False, "think": False,
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 700},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        env = json.loads(r.read())
    return ((env.get("message") or {}).get("content") or "").strip()


def check_golden(pairs: list[dict], corpus: Path) -> int:
    """Every expected answer must be present on the page it cites. A golden set
    that cannot be verified against the source is not ground truth."""
    import fitz
    books = {
        "Professional Chef": "The Professional Chef - The Culinary Institute of America.pdf",
        "Larousse": "Le Grand Larousse gastronomique, 6ème édition (Joël Robuchon).pdf",
    }
    cache: dict[str, list[str]] = {}
    bad = 0
    for p in pairs:
        if not p.get("page"):
            continue
        name = books[p["book"]]
        if name not in cache:
            d = fitz.open(corpus / name)
            cache[name] = [fold(d[i].get_text()) for i in range(d.page_count)]
            d.close()
        page = cache[name][p["page"] - 1]
        missing = [g for g in p["keys"] if not any(fold(a) in page for a in g)]
        if missing:
            bad += 1
            print(f"  NON VERIFICATO {p['id']} p.{p['page']}: gruppi assenti {missing}")
    ok = sum(1 for p in pairs if p.get("page")) - bad
    print(f"ancoraggio: {ok}/{sum(1 for p in pairs if p.get('page'))} coppie verificate "
          f"sulla pagina citata, {bad} non verificate")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("../eval/q2-golden-v1.json"))
    ap.add_argument("--corpus", type=Path, default=Path("../pilot"))
    ap.add_argument("--answerer", default="deepseek-v4-flash:cloud")
    ap.add_argument("--judge", default="qwen3.5:cloud")
    ap.add_argument("--domain", default="pilot")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--check", action="store_true", help="verify the set, do not measure")
    ap.add_argument("--json", type=Path, default=Path("../evidence/T83/q2.json"))
    args = ap.parse_args()

    pairs = json.loads(args.golden.read_text())["pairs"]
    if args.check:
        return check_golden(pairs, args.corpus)

    assert args.answerer.split(":")[0] != args.judge.split(":")[0], \
        "rispondente e giudice devono essere modelli diversi (maker/checker)"

    from graphify_ent.embed import Embedder
    from graphify_ent.loader import Neo4jLoader
    from graphify_ent.retrieval import HybridRetriever

    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    embedder.encode(["warm up"])
    enc = lambda q: embedder.encode([q])[0]  # noqa: E731

    rows = []
    try:
        for p in pairs:
            t0 = time.perf_counter()
            res = retriever.query(p["query"], embed_fn=enc,
                                  channels=("vector", "fulltext", "graph"),
                                  hops=1, domain=args.domain)
            refused = bool(res.refused) or not res.hits
            answer, context = "", ""
            if not refused:
                ids = [h.node_id for h in res.hits[: args.top]]
                props = retriever.hydrate(ids)
                parts = []
                for nid in ids:
                    d = props.get(nid) or {}
                    text = (d.get("passage") or d.get("text_excerpt") or "")[:1200]
                    if text:
                        parts.append(f"[{d.get('source_file','')} {d.get('source_location','')}]"
                                     f"\n{text}")
                context = "\n\n".join(parts)
                answer = call(args.answerer, ANSWER_SYSTEM,
                              f"PASSAGGI:\n{context}\n\nDOMANDA: {p['query']}")
            said_no = refused or "NON_SUPPORTATO" in answer.upper()
            ms = (time.perf_counter() - t0) * 1000

            if p["kind"] == "unanswerable":
                mech = said_no
            else:
                mech = (not said_no) and matches(answer, p["keys"])

            verdict = call(args.judge, JUDGE_SYSTEM,
                           f"DOMANDA: {p['query']}\nRISPOSTA ATTESA: {p['expected']}\n"
                           f"RISPOSTA DEL SISTEMA: {answer or '(rifiuto esplicito)'}")
            verdict = re.sub(r"[^A-ZÀ-Ü]", "", verdict.upper())[:11] or "?"
            judged = (verdict == "RIFIUTO") if p["kind"] == "unanswerable" \
                else (verdict == "CORRETTA")

            rows.append({"id": p["id"], "kind": p["kind"], "lang": p["query_lang"],
                         "acronym": p["acronym"], "book": p["book"], "page": p["page"],
                         "query": p["query"], "expected": p["expected"],
                         "answer": answer, "refused": said_no,
                         "mechanical": mech, "judge": verdict, "judged": judged,
                         "ms": round(ms, 1)})
            print(f"  {'OK ' if mech else 'NO '} {p['id']} [{p['kind'][:12]:<12} "
                  f"{p['query_lang']}] giudice={verdict:<11} {ms:6.0f} ms  "
                  f"{(answer or 'RIFIUTO')[:70]}", flush=True)
    finally:
        loader.close()

    def pct(sel, field):
        s = [r for r in rows if sel(r)]
        return round(100 * sum(r[field] for r in s) / len(s), 1) if s else None

    ans = [r for r in rows if r["kind"] != "unanswerable"]
    report = {
        "golden_set": str(args.golden), "pairs": len(rows),
        "answerer": args.answerer, "judge": args.judge,
        "q2_mechanical_pct": pct(lambda r: True, "mechanical"),
        "q2_judged_pct": pct(lambda r: True, "judged"),
        "q2_reported_pct": min(pct(lambda r: True, "mechanical"),
                               pct(lambda r: True, "judged")),
        "target_g3": 92.0,
        "by_kind": {k: {"n": sum(1 for r in rows if r["kind"] == k),
                        "mechanical_pct": pct(lambda r, k=k: r["kind"] == k, "mechanical"),
                        "judged_pct": pct(lambda r, k=k: r["kind"] == k, "judged")}
                    for k in ("monolingual", "cross_language", "unanswerable")},
        "acronym_pct": pct(lambda r: r["acronym"], "mechanical"),
        "by_lang": {l: pct(lambda r, l=l: r["lang"] == l, "mechanical")
                    for l in ("en", "fr", "it")},
        "answered_when_answerable": sum(1 for r in ans if not r["refused"]),
        "wrong_refusals": sum(1 for r in ans if r["refused"]),
        "latency_ms_mean": round(statistics.mean(r["ms"] for r in rows), 1),
    }
    report["meets_g3"] = report["q2_reported_pct"] >= 92.0
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"report": report, "rows": rows},
                                    indent=1, ensure_ascii=False))
    print("\n" + json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
