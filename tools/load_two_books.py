#!/usr/bin/env python3
"""T74 — load the two-book corpus into the ENTERPRIPHY graph, timed.

Mirrors rebuild_index.py steps 3-6 for a two-book corpus: the pages layer was
built per book and the concept graph assembled from the DeepSeek checkpoint, so
this only loads, enriches, embeds and self-checks — each step timed, because
the load half of "performance" is measured here and not estimated.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

EV = Path("../evidence/T74")
STEPS: dict[str, float] = {}


def timed(label):
    def deco(fn):
        def wrap(*a, **k):
            t0 = time.perf_counter()
            out = fn(*a, **k)
            STEPS[label] = round(time.perf_counter() - t0, 1)
            print(f"   [{label}: {STEPS[label]}s]", flush=True)
            return out
        return wrap
    return deco


@timed("load_graph")
def load_graph():
    from graphify_ent.loader import Neo4jLoader
    loader = Neo4jLoader()
    print(f"target: {loader.uri}", flush=True)
    assert loader.uri.endswith("7688"), "rifiuto: non è l'istanza ENTERPRIPHY (7688)"
    with loader._session() as s:
        s.run("DROP INDEX entity_embedding IF EXISTS")
        s.run("DROP INDEX entity_text IF EXISTS")
    loader.wipe()
    a = loader.load(EV / "corpus-graph.json", domain="pilot")
    b = loader.load(EV / "pages-tpc.json", domain="pilot")
    c = loader.load(EV / "pages-larousse.json", domain="pilot")
    loader.close()
    stats = {"concepts": a.as_dict(), "pages_tpc": b.as_dict(),
             "pages_larousse": c.as_dict()}
    print(json.dumps({k: {kk: v[kk] for kk in ("nodes_written", "db_edges")
                          if kk in v} for k, v in stats.items()}, indent=2))
    return stats


@timed("enrich")
def enrich():
    proc = subprocess.run(
        [sys.executable, "tools/enrich_passages.py", "--corpus", "../input",
         "--checkpoints", str(EV / "slices.jsonl"), "--link",
         "--json", str(EV / "enrich-stats.json")],
        capture_output=True, text=True)
    print(proc.stdout[-1200:])
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
        raise SystemExit("enrich fallito")


@timed("embed")
def embed():
    proc = subprocess.run([sys.executable, "-m", "graphify_ent.embed"],
                          capture_output=True, text=True)
    print(proc.stdout[-600:])
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
        raise SystemExit("embed fallito")


@timed("sanity")
def sanity():
    from graphify_ent.embed import Embedder
    from graphify_ent.loader import Neo4jLoader
    from graphify_ent.retrieval import HybridRetriever
    loader = Neo4jLoader()
    r = HybridRetriever(loader)
    e = Embedder()
    e.encode(["warm up"])
    enc = lambda q: e.encode([q])[0]  # noqa: E731
    checks = {
        "in-corpus EN": not r.query("Bechamel Sauce white roux", embed_fn=enc,
                                    channels=("vector", "fulltext", "graph"),
                                    hops=1, domain="pilot").refused,
        "in-corpus FR": not r.query("biscuit Joconde amandes", embed_fn=enc,
                                    channels=("vector", "fulltext", "graph"),
                                    hops=1, domain="pilot").refused,
        "out-of-corpus refused": r.query(
            "kubernetes ingress controller tls termination", embed_fn=enc,
            channels=("vector", "fulltext", "graph"), hops=1,
            domain="pilot").refused,
    }
    loader.close()
    for k, v in checks.items():
        print(f"   {'OK ' if v else 'NO '} {k}", flush=True)
    return checks


def main() -> int:
    t0 = time.perf_counter()
    load_stats = load_graph()
    enrich()
    embed()
    checks = sanity()
    total = round(time.perf_counter() - t0, 1)
    report = {"steps_seconds": STEPS, "total_seconds": total,
              "load": load_stats, "checks": checks}
    (EV / "load-report.json").write_text(json.dumps(report, indent=2,
                                                    ensure_ascii=False))
    print(f"\ntotale {total}s -> {EV / 'load-report.json'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
