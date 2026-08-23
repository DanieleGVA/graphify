#!/usr/bin/env python3
"""Can the system settle a documentary claim, and how much faster than reading?

This is the task the Mornay check exposed: someone hands you a recipe card that
says "reference followed: Escoffier", and you must decide whether each figure on
it actually matches the source. Retrieval that returns the right *topic* does
not settle anything — the answer is a number in a sentence, so the system has to
hand back the sentence.

Claims are generated mechanically from the source PDFs, never from either
extraction: find a numbered recipe heading, then take a sentence inside it that
carries a quantity. The query is what a cook would type (the dish name plus a
couple of content words); success is that some passage in the top ten contains
that sentence, verbatim up to whitespace. So the metric is not "did it find
something related" but "did it produce the evidence a person needs to adjudicate".

The baseline is what the alternative actually is with no system: open the books
and scan them. Both arms answer the same claims on the same machine, so the
speed-up is a like-for-like measurement rather than a comparison against an
imagined manual workflow.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import fitz

from graphify_ent.embed import Embedder
from graphify_ent.loader import Neo4jLoader
from graphify_ent.retrieval import HybridRetriever

#: "\n131 \nSauce Mornay—Mornay Sauce\n" — a recipe number then its heading.
_HEADING = re.compile(r"\n\s*(\d{2,4})\s*\n([A-ZÀ-Ü][^\n]{6,58})\n")
#: a sentence carrying a weight/volume, i.e. the kind of fact a card gets wrong
_QUANTITY = re.compile(r"[^.\n]{0,90}\b\d{1,4}\s*(?:g|kg|ml|dl|litres?|oz|lb)\b[^.\n]{0,90}")
_NOISE = re.compile(r"[^A-Za-zÀ-ÿ0-9 ,;()\-—/'’.]")


#: PDF text keeps typographic ligatures; "fish" arrives as "ﬁsh" and, once the
#: glyph is lost, as "  sh". Fold them before anything compares strings.
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
              "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}


def defold(s: str) -> str:
    for k, v in _LIGATURES.items():
        s = s.replace(k, v)
    return s


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", defold(s or "")).strip().lower()


#: Structural furniture, not the name of a preparation. The CIA lays its
#: ingredients out under "BASIC FORMULA" headers, so a structural heading
#: finder picks those up and produces claims like "BASIC FORMULA additional
#: bones" — which name nothing and which no system, or person, could settle.
_STRUCTURAL = re.compile(
    r"^(basic formula|ingredients?|method|preparation|notes?|yield|makes|"
    r"variations?|serves|directions|procedure|equipment|chef.s note)\b", re.I)
#: OCR damage: 4+ consecutive consonants, or a text more punctuation than word.
_GARBLED = re.compile(r"[bcdfghjklmnpqrstvwxz]{4,}", re.I)
_LEAD_STOPWORD = re.compile(
    r"^(the|a|an|and|or|of|for|with|to|in|on|at|by|from|as|if|when|while|"
    r"depending|using|about|after|before|le|la|les|des|du|il|lo|gli)\b", re.I)


def readable(s: str) -> bool:
    """Is this text legible enough to be a test item at all?

    Two of the six books are damaged — one scan, one font encoding — and their
    "facts" come out as `S 'WES 4`. Grading a retrieval system on whether it can
    find unreadable strings measures the scan, not the system, so such claims are
    rejected before either arm sees them. The rejection count is reported.
    """
    if not s:
        return False
    letters = sum(c.isalpha() or c.isspace() or c.isdigit() for c in s)
    if letters / len(s) < 0.82:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", s)
    if len(words) < 5:
        return False
    return sum(1 for w in words if _GARBLED.search(w)) / len(words) <= 0.15


def names_something(title: str) -> bool:
    """A claim must be *about* a named preparation, not a table header."""
    if not title or _STRUCTURAL.match(title) or _LEAD_STOPWORD.match(title):
        return False
    # A heading is not a sentence. Anything carrying mid-string punctuation or
    # a lost-ligature gap ("Clari cation") is a fragment the page broke, and a
    # query built from it identifies nothing.
    if re.search(r"[.;:]\s", title) or re.search(r"\b[a-z]{1,3} [a-z]{3,}ation\b", title):
        return False
    words = title.split()
    return 2 <= len(words) <= 7 and not title.rstrip().endswith((",", "and", "or"))


def clean_title(t: str) -> str:
    t = _NOISE.sub(" ", t)
    t = re.split(r"[—–]| or | ou ", t)[0]
    return re.sub(r"\s+", " ", t).strip()


def heading_before(lines: list[str], idx: int) -> str:
    """Nearest preceding line that reads like a dish name.

    Escoffier numbers its recipes, the CIA and the Larousse do not. Keying on
    Escoffier's numbering would have limited the whole benchmark to one book —
    it did, on the first run — so the heading is found structurally instead:
    a short line, no closing punctuation, not itself a quantity.
    """
    for j in range(idx - 1, max(-1, idx - 9), -1):
        s = lines[j].strip()
        if not (6 <= len(s) <= 60):
            continue
        if s.endswith((".", ",", ";", ":")) or re.search(r"\d\s*(g|oz|ml|lb)\b", s):
            continue
        if not re.match(r"[A-Za-zÀ-ÿ]", s):
            continue
        if sum(c.isdigit() for c in s) > 3:
            continue
        return clean_title(s)
    return ""


def build_claims(pdf: Path, want: int, stride: int, rejected: list) -> list[dict]:
    doc = fitz.open(pdf)
    out: list[dict] = []
    for i in range(0, doc.page_count, stride):
        if len(out) >= want:
            break
        text = doc[i].get_text()
        lines = text.split("\n")
        per_page = 0
        for li, line in enumerate(lines):
            if per_page >= 2 or len(out) >= want:
                break
            q = _QUANTITY.search(line)
            if not q:
                continue
            fact = re.sub(r"\s+", " ", " ".join(lines[li:li + 2])).strip()
            fact = fact[:150]
            if len(fact) < 40 or len(fact.split()) < 7:
                continue
            title = heading_before(lines, li)
            if not names_something(title) or not readable(fact):
                rejected[0] += 1
                continue
            context = re.findall(r"[A-Za-zÀ-ÿ]{5,}", " ".join(lines[li:li + 6]))[:4]
            out.append({
                "book": pdf.name, "page": i + 1, "recipe": title,
                "query": f"{title} {' '.join(context)}".strip(),
                "fact": fact,
            })
            per_page += 1
    doc.close()
    return out


def contains(fact: str, text: str) -> bool:
    """Whitespace-insensitive containment — PDF text breaks lines mid-sentence."""
    if not text:
        return False
    pattern = r"\s+".join(re.escape(w) for w in norm(fact).split())
    return re.search(pattern, text, re.I) is not None


def baseline_scan(fact: str, pdfs: list[Path], cache: dict | None) -> tuple[bool, float]:
    """What you do without the system: open the books and read until you find it.

    Two baselines, because the difference between them is the whole argument.

      cold (cache=None) — open the PDFs and extract their text now. This is the
        real alternative: nobody keeps 337 MB of books parsed in RAM between
        questions, and it is what the manual Mornay check actually cost.

      warm (cache given) — the corpus already parsed in memory, so the scan is
        a plain regex over 13 M characters. Reported as the conservative floor:
        it credits the naive approach with an index it never built, while the
        system's index was built once, offline, and is the thing being paid for.
    """
    t0 = time.perf_counter()
    pattern = r"\s+".join(re.escape(w) for w in norm(fact).split())
    rx = re.compile(pattern, re.I)
    for pdf in pdfs:
        if cache is None:
            doc = fitz.open(pdf)
            pages = [doc[i].get_text() for i in range(doc.page_count)]
            doc.close()
        else:
            if str(pdf) not in cache:
                doc = fitz.open(pdf)
                cache[str(pdf)] = [doc[i].get_text() for i in range(doc.page_count)]
                doc.close()
            pages = cache[str(pdf)]
        for page in pages:
            if rx.search(page):
                return True, (time.perf_counter() - t0) * 1000
    return False, (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("../pilot"))
    ap.add_argument("--per-book", type=int, default=8)
    ap.add_argument("--stride", type=int, default=37)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cold-sample", type=int, default=10,
                    help="how many claims to time against a cold scan; the cost is "
                         "dominated by parsing 337 MB of PDF and is near-constant, "
                         "so timing every claim only measures the slow method")
    ap.add_argument("--claims", type=Path, default=Path("../eval/verify-claims-v1.json"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--json", type=Path, default=Path("../evidence/T72/bench-verify.json"))
    args = ap.parse_args()

    pdfs = sorted(args.corpus.glob("*.pdf"))
    if args.rebuild or not args.claims.exists():
        claims: list[dict] = []
        rejected = [0]
        for pdf in pdfs:
            claims += build_claims(pdf, args.per_book, args.stride, rejected)
        args.claims.parent.mkdir(parents=True, exist_ok=True)
        args.claims.write_text(json.dumps(
            {"version": "verify-v1",
             "provenance": "recipe headings and quantity sentences taken from the "
                           "source PDFs; no extraction contributed to the claims",
             "rejected_unusable": rejected[0],
             "rejection_rule": "a claim must name a preparation (not a table "
                               "header) and its fact must be legible; damaged-scan "
                               "text measures the scan, not the system",
             "claims": claims}, indent=2, ensure_ascii=False))
        print(f"generate {len(claims)} affermazioni -> {args.claims}")
    claims = json.loads(args.claims.read_text())["claims"]
    print(f"affermazioni da verificare: {len(claims)}\n")

    loader = Neo4jLoader()
    retriever = HybridRetriever(loader)
    embedder = Embedder()
    embedder.encode(["warm up"])                 # model load stays out of the timing

    rows = []
    cold_samples: list[float] = []
    scan_cache: dict = {}
    try:
        for c in claims:
            t0 = time.perf_counter()
            res = retriever.query(c["query"], embedding=embedder.encode([c["query"]])[0],
                                  channels=("vector", "fulltext", "graph"), hops=1,
                                  domain="pilot")
            ids = [h.node_id for h in res.hits[: args.top]]
            with loader._session() as s:
                props = {r["id"]: r for r in s.run(
                    "MATCH (n:Entity) WHERE n.id IN $i RETURN n.id AS id, "
                    "n.passage AS p, n.text_excerpt AS x, n.source_file AS f, "
                    "n.source_location AS loc", i=ids)}
            ok = False
            where = None
            for nid in ids:
                p = props.get(nid) or {}
                if contains(c["fact"], (p.get("p") or "") + " " + (p.get("x") or "")):
                    ok, where = True, f"{p.get('f','')[:14]} {p.get('loc','')}"
                    break
            sys_ms = (time.perf_counter() - t0) * 1000
            warm_ok, warm_ms = baseline_scan(c["fact"], pdfs, scan_cache)
            if len(cold_samples) < args.cold_sample:
                cold_ok, cold_ms = baseline_scan(c["fact"], pdfs, None)
                cold_samples.append(cold_ms)
            else:
                cold_ok, cold_ms = warm_ok, statistics.mean(cold_samples)
            rows.append({**c, "system_found": ok, "system_ms": round(sys_ms, 1),
                         "system_where": where, "baseline_found": warm_ok or cold_ok,
                         "warm_ms": round(warm_ms, 1), "cold_ms": round(cold_ms, 1)})
            print(f"  {'OK ' if ok else 'NO '} sistema {sys_ms:7.0f} ms | "
                  f"lettura a freddo {cold_ms:8.0f} ms | in memoria {warm_ms:7.0f} ms  "
                  f"{c['recipe'][:30]:<32} {where or ''}", flush=True)
    finally:
        loader.close()

    found = [r for r in rows if r["system_found"]]
    reachable = [r for r in rows if r["baseline_found"]]
    sys_total = sum(r["system_ms"] for r in rows)
    cold_total = sum(r["cold_ms"] for r in rows)
    warm_total = sum(r["warm_ms"] for r in rows)
    report = {
        "claims": len(rows),
        "reachable_in_corpus": len(reachable),
        "accuracy_pct": round(100 * len(found) / max(len(reachable), 1), 2),
        "system_ms_mean": round(statistics.mean([r["system_ms"] for r in rows]), 1),
        "system_ms_p95": round(sorted(r["system_ms"] for r in rows)[int(0.95 * len(rows)) - 1], 1),
        "cold_ms_mean": round(statistics.mean([r["cold_ms"] for r in rows]), 1),
        "warm_ms_mean": round(statistics.mean([r["warm_ms"] for r in rows]), 1),
        "speedup_x": round(cold_total / sys_total, 1) if sys_total else None,
        "speedup_vs_in_memory_x": round(warm_total / sys_total, 1) if sys_total else None,
        "target": {"accuracy_pct": 95.0, "speedup_x": 20.0},
    }
    report["meets_goal"] = bool(
        report["accuracy_pct"] >= 95.0 and (report["speedup_x"] or 0) >= 20.0)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"report": report, "rows": rows},
                                    indent=2, ensure_ascii=False))
    print("\n" + json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
