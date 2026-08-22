#!/usr/bin/env python3
"""Phase 2 evidence — benchmark the new loader against the autocommit baseline.

The baseline replicates `graphify/export.py:push_to_neo4j` exactly: one
autocommit `session.run` per node and per edge, and **no uniqueness constraint**
(that is the upstream state — the constraint is what turns each MERGE from a
label scan into an index lookup).

Usage: python tools/bench_loader.py [--n 20000] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from graphify_ent.loader import Neo4jLoader, safe_label, safe_rel


def make_graph(n: int, tmp: Path) -> Path:
    nodes = [
        {"id": f"n{i}", "label": f"Node {i}", "file_type": "document",
         "source_file": f"doc{i % 50}.pdf", "text_excerpt": f"excerpt for node {i}"}
        for i in range(n)
    ]
    edges = [
        {"source": f"n{i % n}", "target": f"n{(i * 7 + 1) % n}", "relation": "references",
         "confidence": "EXTRACTED", "weight": 1.0}
        for i in range(2 * n)
    ]
    p = tmp / "bench_graph.json"
    p.write_text(json.dumps({"nodes": nodes, "edges": edges}))
    return p


def bench_autocommit(loader: Neo4jLoader, graph: Path, n_cap: int) -> dict:
    """Upstream path: per-element autocommit, no constraints."""
    data = json.loads(graph.read_text())
    nodes = data["nodes"][:n_cap]
    ids = {x["id"] for x in nodes}
    edges = [e for e in data["edges"] if e["source"] in ids and e["target"] in ids][:n_cap]

    loader.wipe()
    # Deliberately NO apply_schema(): the upstream exporter never created one.
    with loader._session() as s:
        t0 = time.perf_counter()
        for node in nodes:
            props = {k: v for k, v in node.items() if isinstance(v, (str, int, float, bool))}
            ftype = safe_label(str(node.get("file_type", "Entity")).capitalize())
            s.run(f"MERGE (n:{ftype} {{id: $id}}) SET n += $props", id=node["id"], props=props)
        t_nodes = time.perf_counter() - t0

        t0 = time.perf_counter()
        for e in edges:
            props = {k: v for k, v in e.items() if isinstance(v, (str, int, float, bool))}
            rel = safe_rel(str(e.get("relation", "RELATED_TO")))
            s.run(
                f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) MERGE (a)-[r:{rel}]->(b) SET r += $props",
                src=e["source"], tgt=e["target"], props=props,
            )
        t_edges = time.perf_counter() - t0

    return {
        "nodes": len(nodes), "edges": len(edges),
        "nodes_seconds": round(t_nodes, 3), "edges_seconds": round(t_edges, 3),
        "nodes_per_s": round(len(nodes) / t_nodes, 1) if t_nodes else 0,
        "edges_per_s": round(len(edges) / t_edges, 1) if t_edges else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000, help="nodes for the new loader")
    ap.add_argument("--baseline-n", type=int, default=2_000,
                    help="elements for the O(n^2) baseline (kept small on purpose)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    graph = make_graph(args.n, tmp)

    loader = Neo4jLoader()
    try:
        base = bench_autocommit(loader, graph, args.baseline_n)
        print("baseline (autocommit, no constraint):", json.dumps(base, indent=2))

        loader.wipe()
        loader.apply_schema()
        stats = loader.load(graph, domain="bench")
        new = stats.as_dict()
        print("new loader (UNWIND batches, constraint-first):", json.dumps(new, indent=2))

        report = {
            "baseline": base,
            "new_loader": new,
            "speedup_nodes_x": round(new["nodes_per_s"] / base["nodes_per_s"], 1)
            if base["nodes_per_s"] else None,
            "speedup_edges_x": round(new["edges_per_s"] / base["edges_per_s"], 1)
            if base["edges_per_s"] else None,
            "acceptance": {
                "nodes_per_s_target": 10_000,
                "edges_per_s_target": 20_000,
                "nodes_pass": new["nodes_per_s"] > 10_000,
                "edges_pass": new["edges_per_s"] > 20_000,
            },
        }
        print("\n" + json.dumps(report["speedup_nodes_x"] and report["acceptance"], indent=2))
        print(f"speedup: nodes x{report['speedup_nodes_x']}, edges x{report['speedup_edges_x']}")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2))
        return 0
    finally:
        loader.close()


if __name__ == "__main__":
    raise SystemExit(main())
