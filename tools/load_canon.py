#!/usr/bin/env python3
"""T90 — load the twelve-book library into its own domain, timed.

`canon_library` is additive: the two-book `pilot` graph stays exactly where it
is, in the same database, separated by the `domain` property that every tool
already filters on. Nothing is dropped, so the approved benchmarks keep their
ground under them and a reader chooses which corpus to ask.

Two consequences of scale are handled here rather than discovered later:

  * **The vector index is dropped for the load and rebuilt after.** Measured on
    the two-book graph: with the HNSW index online, every insertion updates the
    graph and the writer sat at 0.2% CPU waiting on the database.
  * **Only the new domain is re-embedded.** The pilot nodes already carry
    BGE-m3 vectors of the right width; re-encoding them would cost an hour to
    produce identical numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

EV = Path("../evidence/T90")
DOMAIN = "canon_library"
STEPS: dict[str, float] = {}


def step(label: str):
    def deco(fn):
        def wrap(*a, **k):
            t0 = time.perf_counter()
            print(f"\n▸ {label}", flush=True)
            out = fn(*a, **k)
            STEPS[label] = round(time.perf_counter() - t0, 1)
            print(f"   ({STEPS[label]}s)", flush=True)
            return out
        return wrap
    return deco


@step("1/5 caricamento concetti e pagine")
def load_graph():
    from graphify_ent.loader import Neo4jLoader
    loader = Neo4jLoader()
    assert loader.uri.endswith("7688"), "rifiuto: non è l'istanza ENTERPRIPHY (7688)"
    with loader._session() as s:
        before = s.run("MATCH (n:Entity {domain:$d}) RETURN count(n) AS c",
                       d="pilot").single()["c"]
        s.run("DROP INDEX entity_embedding IF EXISTS")
    print(f"   dominio pilot preservato: {before:,} nodi")
    stats = {"concepts": loader.load(EV / "corpus-graph.json", domain=DOMAIN).as_dict()}
    pages = sorted(glob.glob(str(EV / "pages" / "*.json")))
    written = 0
    for p in pages:
        if p.endswith("_summary.json"):
            continue
        r = loader.load(Path(p), domain=DOMAIN).as_dict()
        written += r["nodes_written"]
    stats["pages"] = {"files": len(pages) - 1, "nodes_written": written}
    with loader._session() as s:
        after = s.run("MATCH (n:Entity {domain:$d}) RETURN count(n) AS c",
                      d="pilot").single()["c"]
    assert after == before, f"il dominio pilot è cambiato: {before} -> {after}"
    loader.close()
    print(f"   concetti {stats['concepts']['nodes_written']:,} · "
          f"pagine {written:,} · pilot intatto")
    return stats


@step("2/5 passaggi e collegamenti concetto→pagina")
def enrich():
    r = subprocess.run(
        [sys.executable, "tools/enrich_passages.py", "--corpus", "../canon_library",
         "--checkpoints", str(EV / "slices.jsonl"), "--link",
         "--json", str(EV / "enrich-stats.json")],
        capture_output=True, text=True)
    print(r.stdout[-700:])
    if r.returncode != 0:
        print(r.stderr[-1200:])
        raise SystemExit("enrich fallito")


@step("3/5 vettori (solo il nuovo dominio)")
def embed():
    r = subprocess.run([sys.executable, "-m", "graphify_ent.embed", "--batch", "256",
                        "--ledger", "cost-embed-canon.json"],
                       capture_output=True, text=True)
    print(r.stdout[-500:])
    if r.returncode != 0:
        print(r.stderr[-1200:])
        raise SystemExit("embed fallito")


@step("4/5 ricostruzione indice vettoriale")
def reindex():
    from graphify_ent.loader import Neo4jLoader
    loader = Neo4jLoader()
    loader.apply_schema()
    with loader._session() as s:
        s.run("CALL db.awaitIndex('entity_embedding', 1800)")
        state = list(s.run("SHOW INDEXES YIELD name,state "
                           "WHERE name='entity_embedding' RETURN state"))[0]["state"]
    loader.close()
    print(f"   indice: {state}")
    return state


@step("5/5 verifica")
def verify():
    from graphify_ent.embed import Embedder
    from graphify_ent.loader import Neo4jLoader
    from graphify_ent.retrieval import HybridRetriever
    loader = Neo4jLoader()
    r = HybridRetriever(loader)
    e = Embedder()
    e.encode(["warm up"])
    enc = lambda q: e.encode([q])[0]  # noqa: E731
    with loader._session() as s:
        counts = {row["d"]: row["c"] for row in s.run(
            "MATCH (n:Entity) RETURN n.domain AS d, count(*) AS c")}
        books = s.run("MATCH (n:Entity {domain:$d}) RETURN count(DISTINCT n.source_file) AS b",
                      d=DOMAIN).single()["b"]
    checks = {
        "risposta dal nuovo dominio": not r.query(
            "sauce espagnole preparation", embed_fn=enc, hops=1, domain=DOMAIN).refused,
        "domanda estranea rifiutata": r.query(
            "kubernetes ingress controller tls termination", embed_fn=enc,
            hops=1, domain=DOMAIN).refused,
        "dominio pilot ancora interrogabile": not r.query(
            "Bechamel Sauce white roux", embed_fn=enc, hops=1, domain="pilot").refused,
    }
    loader.close()
    for k, v in checks.items():
        print(f"   {'OK ' if v else 'NO '} {k}")
    print(f"   nodi per dominio: {counts} · libri distinti: {books}")
    return {"checks": checks, "counts": counts, "books": books}


def main() -> int:
    t0 = time.perf_counter()
    load_stats = load_graph()
    enrich()
    embed()
    index_state = reindex()
    v = verify()
    report = {"domain": DOMAIN, "steps_seconds": STEPS,
              "total_seconds": round(time.perf_counter() - t0, 1),
              "load": load_stats, "index_state": index_state, **v}
    (EV / "load-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\ntotale {report['total_seconds']}s -> {EV / 'load-report.json'}")
    return 0 if all(v["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
