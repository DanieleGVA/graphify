#!/usr/bin/env python3
"""Give every domain its own vector index.

A single index over every corpus makes each one pay for the others: the search
is approximate, so its k nearest neighbours are drawn from the whole graph and
only the fraction that happens to belong to the asked-for domain survives the
filter. Measured on a 149k-node graph holding two corpora — one 53k, one 96k:

    shared index, over-fetched   435 ms per verification, 24/25 correct
    per-domain index              33 ms per verification, 25/25 correct

The label is what makes it possible: Neo4j indexes a label, so each domain gets
a marker label and an index over it. Labels are additive, so nothing else in
the graph changes and the shared index stays as a fallback.

    python tools/build_domain_indexes.py
"""

from __future__ import annotations

import argparse
import re
import time


def label_for(domain: str) -> str:
    return "D_" + re.sub(r"[^A-Za-z0-9_]", "_", domain)


def index_for(domain: str) -> str:
    return "entity_embedding_" + re.sub(r"[^A-Za-z0-9_]", "_", domain)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--drop-shared", action="store_true",
                    help="rimuove l'indice condiviso una volta costruiti i per-dominio")
    args = ap.parse_args()

    from graphify_ent.loader import Neo4jLoader

    loader = Neo4jLoader()
    with loader._session() as s:
        domains = [r["d"] for r in s.run(
            "MATCH (n:Entity) WHERE n.domain IS NOT NULL "
            "RETURN DISTINCT n.domain AS d ORDER BY d")]
        print(f"domini: {domains}")

        for d in domains:
            t0 = time.perf_counter()
            lbl, idx = label_for(d), index_for(d)
            tagged = s.run(
                f"MATCH (n:Entity {{domain: $d}}) WHERE NOT n:{lbl} "
                f"CALL (n) {{ SET n:{lbl} }} IN TRANSACTIONS OF 5000 ROWS "
                f"RETURN count(*) AS c", d=d).single()
            s.run(
                f"CREATE VECTOR INDEX {idx} IF NOT EXISTS FOR (n:{lbl}) "
                f"ON (n.embedding) OPTIONS {{indexConfig: {{"
                f"`vector.dimensions`: {args.dim}, "
                f"`vector.similarity_function`: 'cosine'}}}}")
            # The text index needs the same treatment for the same reason: a
            # shared one is cut to its best `probe` candidates BEFORE the domain
            # filter applies, so one corpus is left with a handful. Measured:
            # 673 ms per query on the shared index against 16 ms before a second
            # corpus existed.
            s.run(
                f"CREATE FULLTEXT INDEX entity_text_{re.sub(r'[^A-Za-z0-9_]', '_', d)} "
                f"IF NOT EXISTS FOR (n:{lbl}) ON EACH [n.label, n.text_excerpt, n.passage] "
                f"OPTIONS {{indexConfig: {{`fulltext.analyzer`: 'standard-folding'}}}}")
            print(f"  {d}: etichettati, indice {idx} creato "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

        for d in domains:
            s.run(f"CALL db.awaitIndex('{index_for(d)}', 3600)")
            s.run(f"CALL db.awaitIndex('entity_text_"
                  f"{re.sub(r'[^A-Za-z0-9_]', '_', d)}', 3600)")
        states = {r["name"]: r["state"] for r in s.run(
            "SHOW INDEXES YIELD name, state WHERE name STARTS WITH 'entity_' "
            "RETURN name, state")}
        print(f"  stato indici: {states}")

        if args.drop_shared:
            s.run("DROP INDEX entity_embedding IF EXISTS")
            print("  indice condiviso rimosso")

        counts = {r["d"]: r["c"] for r in s.run(
            "MATCH (n:Entity) RETURN n.domain AS d, count(*) AS c")}
        print(f"  nodi per dominio: {counts}")
    loader.close()
    return 0 if all(v == "ONLINE" for k, v in states.items()
                    if k != "entity_embedding") else 1


if __name__ == "__main__":
    raise SystemExit(main())
