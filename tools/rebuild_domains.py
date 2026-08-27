#!/usr/bin/env python3
"""Rebuild every domain from its artifacts, under the (id, domain) identity.

Needed once, because the graph on disk was built when identity was `id` alone
and a second corpus sharing two source books adopted the first's nodes. Both
corpora are fully reconstructible — the extraction checkpoints and the page
layers are on disk — so the honest repair is to rebuild rather than to patch
rows in place and hope.

Order matters and is not negotiable:
  schema (composite constraint) → load → enrich → embed → vector index → check
The vector index is created LAST: with it online every insert updates the HNSW
graph and the loader waits on the database instead of working.
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

DOMAINS = {
    "pilot": {
        "corpus": Path("../pilot"),
        "concepts": [Path("../evidence/T74/corpus-graph.json")],
        "pages": [Path("../evidence/T74/pages-tpc.json"),
                  Path("../evidence/T74/pages-larousse.json")],
        "checkpoints": [Path("../evidence/T74/slices.jsonl")],
    },
    "canon_library": {
        "corpus": Path("../canon_library"),
        "concepts": [Path("../evidence/T90/corpus-graph.json")],
        "pages": sorted(p for p in Path("../evidence/T90/pages").glob("*.json")
                        if p.name != "_summary.json"),
        "checkpoints": [Path("../evidence/T90/slices.jsonl")],
    },
}
REPORT = Path("../evidence/T90/rebuild.json")


def main() -> int:
    from graphify_ent.loader import Neo4jLoader

    t_all = time.perf_counter()
    steps: dict[str, float] = {}
    loader = Neo4jLoader()
    assert loader.uri.endswith("7688"), "rifiuto: non è l'istanza ENTERPRIPHY (7688)"

    print("▸ schema: identità composta (id, domain)")
    with loader._session() as s:
        s.run("DROP INDEX entity_embedding IF EXISTS")
        s.run("DROP INDEX entity_text IF EXISTS")
        s.run("DROP CONSTRAINT entity_id IF EXISTS")
    loader.wipe()
    applied = loader.apply_schema()
    print(f"   {len(applied)} istruzioni applicate, grafo azzerato")

    loaded = {}
    for name, cfg in DOMAINS.items():
        t0 = time.perf_counter()
        print(f"\n▸ dominio {name}")
        concepts = pages = 0
        for f in cfg["concepts"]:
            concepts += loader.load(f, domain=name).as_dict()["nodes_written"]
        for f in cfg["pages"]:
            pages += loader.load(Path(f), domain=name).as_dict()["nodes_written"]
        loaded[name] = {"concepts": concepts, "pages": pages}
        steps[f"load:{name}"] = round(time.perf_counter() - t0, 1)
        print(f"   concetti {concepts:,} · pagine {pages:,} ({steps[f'load:{name}']}s)")
    loader.close()

    for name, cfg in DOMAINS.items():
        t0 = time.perf_counter()
        print(f"\n▸ arricchimento {name}")
        r = subprocess.run(
            [sys.executable, "tools/enrich_passages.py",
             "--corpus", str(cfg["corpus"]), "--domain", name, "--link",
             "--checkpoints", *[str(c) for c in cfg["checkpoints"]],
             "--json", f"../evidence/T90/enrich-{name}.json"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-800:], r.stderr[-800:])
            raise SystemExit(f"enrich fallito per {name}")
        tail = [l for l in r.stdout.splitlines() if '"' in l][-6:]
        print("   " + " ".join(t.strip() for t in tail)[:200])
        steps[f"enrich:{name}"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    print("\n▸ vettori (tutti i domini)")
    r = subprocess.run([sys.executable, "-m", "graphify_ent.embed",
                        "--batch", "256", "--ledger", "cost-embed-all.json"],
                       capture_output=True, text=True)
    print(r.stdout[-400:])
    if r.returncode != 0:
        print(r.stderr[-1000:])
        raise SystemExit("embed fallito")
    steps["embed"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    print("\n▸ indice vettoriale")
    loader = Neo4jLoader()
    loader.apply_schema()
    with loader._session() as s:
        s.run("CALL db.awaitIndex('entity_embedding', 3600)")
        state = list(s.run("SHOW INDEXES YIELD name,state "
                           "WHERE name='entity_embedding' RETURN state"))[0]["state"]
        counts = {r["d"]: r["c"] for r in s.run(
            "MATCH (n:Entity) RETURN n.domain AS d, count(*) AS c")}
        shared = s.run(
            "MATCH (n:Entity) WITH n.id AS id, count(DISTINCT n.domain) AS d "
            "WHERE d > 1 RETURN count(*) AS c").single()["c"]
        crossed = s.run(
            "MATCH (a:Entity)-[]->(b:Entity) WHERE a.domain <> b.domain "
            "RETURN count(*) AS c").single()["c"]
    steps["index"] = round(time.perf_counter() - t0, 1)
    print(f"   indice {state} · nodi per dominio {counts}")
    print(f"   id presenti in più domini: {shared:,} (attesi: i libri condivisi)")
    print(f"   archi che attraversano i domini: {crossed} (deve essere 0)")

    from graphify_ent.embed import Embedder
    from graphify_ent.retrieval import HybridRetriever
    r = HybridRetriever(loader)
    e = Embedder()
    e.encode(["warm up"])
    enc = lambda q: e.encode([q])[0]  # noqa: E731
    checks = {
        "pilot risponde": not r.query("Bechamel Sauce white roux", embed_fn=enc,
                                      hops=1, domain="pilot").refused,
        "canon_library risponde": not r.query("sauce espagnole", embed_fn=enc,
                                              hops=1, domain="canon_library").refused,
        "estranea rifiutata": r.query("kubernetes ingress controller tls",
                                      embed_fn=enc, hops=1,
                                      domain="canon_library").refused,
        "nessun arco fra domini": crossed == 0,
    }
    loader.close()
    for k, v in checks.items():
        print(f"   {'OK ' if v else 'NO '} {k}")

    report = {"steps_seconds": steps,
              "total_seconds": round(time.perf_counter() - t_all, 1),
              "loaded": loaded, "counts": counts, "index_state": state,
              "ids_in_multiple_domains": shared, "cross_domain_edges": crossed,
              "checks": checks}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\ntotale {report['total_seconds']}s -> {REPORT}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
