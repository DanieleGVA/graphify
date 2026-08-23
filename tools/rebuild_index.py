#!/usr/bin/env python3
"""Rebuild the whole searchable index for one document, from PDF to answers.

Six steps, in the only order they work in. Written down because rebuilding it
by hand invites leaving one out, and the ones easiest to forget — re-embedding
after the passages change, rebuilding the text index after that — fail
silently: the system keeps answering, just from stale vectors.

    python tools/rebuild_index.py "../test_input/The Professional Chef_abstract.pdf"

Extraction is skipped when its checkpoint already exists, so a re-run costs
seconds and no model calls. Pass --extract to force it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(label: str, argv: list[str]) -> None:
    t0 = time.perf_counter()
    print(f"\n▸ {label}", flush=True)
    proc = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(f"passo fallito: {label}")
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-3:]
    for ln in tail:
        print(f"   {ln}")
    print(f"   ({time.perf_counter() - t0:.1f}s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--evidence", type=Path, default=Path("../evidence/T73"))
    ap.add_argument("--domain", default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extract", action="store_true", help="re-run LLM extraction")
    args = ap.parse_args()

    ev = args.evidence
    ev.mkdir(parents=True, exist_ok=True)
    pages, concepts = ev / "pages.json", ev / "concepts.json"
    checkpoint = ev / "slices.jsonl"
    t0 = time.perf_counter()

    run("1/6 strato delle pagine (deterministico)",
        ["tools/build_pages.py", str(args.pdf), "--domain", args.domain,
         "--out", str(pages)])

    if args.extract or not checkpoint.exists():
        run("2/6 estrazione semantica (LLM)",
            ["tools/ingest_corpus.py", "--corpus", str(args.pdf.parent),
             "--workers", str(args.workers), "--checkpoint", str(checkpoint),
             "--out", str(concepts), "--stats", str(ev / "ingest-stats.json")])
    else:
        print(f"\n▸ 2/6 estrazione semantica — saltata, checkpoint presente "
              f"({checkpoint})")

    print("\n▸ 3/6 caricamento nel grafo", flush=True)
    from graphify_ent.loader import Neo4jLoader
    loader = Neo4jLoader()
    with loader._session() as s:
        # The vector index carries a width and the text index an analyzer;
        # CREATE ... IF NOT EXISTS never updates either, so a changed model or
        # analyzer would leave the old one in place and quietly wrong.
        s.run("DROP INDEX entity_embedding IF EXISTS")
        s.run("DROP INDEX entity_text IF EXISTS")
    loader.wipe()
    a = loader.load(concepts, domain=args.domain)
    b = loader.load(pages, domain=args.domain)
    loader.close()
    print(f"   concetti {a.as_dict()['nodes_written']:,} · "
          f"pagine {b.as_dict()['nodes_written']:,} · "
          f"archi {b.as_dict()['db_edges']:,}")

    run("4/6 passaggi e collegamenti concetto→pagina",
        ["tools/enrich_passages.py", "--corpus", str(args.pdf.parent),
         "--checkpoints", str(checkpoint), "--link",
         "--json", str(ev / "enrich-stats.json")])

    run("5/6 vettori", ["-m", "graphify_ent.embed"])

    print("\n▸ 6/6 verifica di funzionamento", flush=True)
    from graphify_ent.embed import Embedder
    from graphify_ent.retrieval import HybridRetriever
    loader = Neo4jLoader()
    r = HybridRetriever(loader)
    e = Embedder()
    e.encode(["warm up"])
    enc = lambda q: e.encode([q])[0]                       # noqa: E731
    checks = {
        "domanda nel corpus": not r.query(
            "Bechamel Sauce white roux", embed_fn=enc,
            channels=("vector", "fulltext", "graph"), hops=1, domain=args.domain).refused,
        "domanda estranea rifiutata": r.query(
            "kubernetes ingress controller tls termination", embed_fn=enc,
            channels=("vector", "fulltext", "graph"), hops=1, domain=args.domain).refused,
    }
    loader.close()
    for k, v in checks.items():
        print(f"   {'OK ' if v else 'NO '} {k}")
    ok = all(checks.values())
    print(f"\ntotale {time.perf_counter() - t0:.1f}s — "
          f"{'indice pronto' if ok else 'INDICE NON SANO'}")
    (ev / "rebuild.json").write_text(json.dumps(
        {"pdf": str(args.pdf), "checks": checks,
         "seconds": round(time.perf_counter() - t0, 1)}, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
