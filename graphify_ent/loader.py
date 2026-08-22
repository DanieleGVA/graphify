"""Phase 2 — Neo4j schema application + bulk loader.

Replaces `graphify/export.py:push_to_neo4j`, which issues one autocommit
`session.run` per node *and* per edge with no uniqueness constraint — so every
MERGE is a label scan and the cost is O(n²) beyond ~100k elements (architecture
doc §1, finding 5). The upstream exporter is left untouched (plan §2); only its
`_safe_rel` / `_safe_label` sanitization discipline is carried over.

This loader:
  1. applies `schema.cypher` idempotently *before* any write,
  2. UNWINDs nodes in batches of 5,000 and edges in batches of 10,000 inside
     explicit transactions, with deadlock retry,
  3. matches edge endpoints by the indexed `id`,
  4. stamps `domain`, `ingested_at`, and reconciles final counts against input.

CLI:
    python -m graphify_ent.loader load graph.json --domain pilot [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "LoadStats",
    "Neo4jLoader",
    "batched",
    "iter_edges",
    "iter_nodes",
    "safe_label",
    "safe_rel",
    "sanitize_props",
]

NODE_BATCH = 5_000
EDGE_BATCH = 10_000
DEADLOCK_RETRIES = 5

_SCHEMA_PATH = Path(__file__).with_name("schema.cypher")
_VALID_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ------------------------------------------------------------- sanitization

def safe_rel(relation: str) -> str:
    """Relationship type, injection-safe (upstream `_safe_rel`)."""
    cleaned = re.sub(
        r"[^A-Z0-9_]", "_", (relation or "").upper().replace(" ", "_").replace("-", "_")
    )
    return cleaned or "RELATED_TO"


def safe_label(label: str) -> str:
    """Node label, injection-safe (upstream `_safe_label`)."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "", label or "")
    return sanitized or "Entity"


def sanitize_props(data: dict) -> dict:
    """Keep primitive-valued properties with Cypher-safe keys.

    Neo4j cannot store nested structures as properties, and a key containing a
    backtick or space would have to be quoted — we drop those instead of
    quoting so no caller can smuggle Cypher through a property name.
    """
    return {
        k: v
        for k, v in (data or {}).items()
        if isinstance(v, (str, int, float, bool)) and _VALID_KEY.match(str(k))
    }


# ------------------------------------------------------------- input parsing

def _load_json(path: Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_nodes(path: Path) -> Iterator[dict]:
    """Yield node dicts from graph.json or a nodes.jsonl staging file."""
    path = Path(path)
    if path.suffix == ".jsonl":
        yield from _iter_jsonl(path)
        return
    data = _load_json(path)
    for n in (data.get("nodes") if isinstance(data, dict) else data) or []:
        yield n


def iter_edges(path: Path) -> Iterator[dict]:
    """Yield edge dicts from graph.json (`edges` or networkx `links`) or JSONL."""
    path = Path(path)
    if path.suffix == ".jsonl":
        yield from _iter_jsonl(path)
        return
    data = _load_json(path)
    if not isinstance(data, dict):
        return
    for e in data.get("edges") or data.get("links") or []:
        yield e


def batched(it: Iterable, size: int) -> Iterator[list]:
    batch: list = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ------------------------------------------------------------------- loading

@dataclass
class LoadStats:
    nodes_written: int = 0
    edges_written: int = 0
    nodes_seconds: float = 0.0
    edges_seconds: float = 0.0
    reconciled: bool = False
    db_nodes: int = 0
    db_edges: int = 0
    skipped_edges: int = 0
    unmatched_edges: int = 0
    per_relation: dict[str, int] = field(default_factory=dict)

    @property
    def nodes_per_s(self) -> float:
        return self.nodes_written / self.nodes_seconds if self.nodes_seconds else 0.0

    @property
    def edges_per_s(self) -> float:
        return self.edges_written / self.edges_seconds if self.edges_seconds else 0.0

    def as_dict(self) -> dict:
        return {
            "nodes_written": self.nodes_written,
            "edges_written": self.edges_written,
            "nodes_per_s": round(self.nodes_per_s, 1),
            "edges_per_s": round(self.edges_per_s, 1),
            "nodes_seconds": round(self.nodes_seconds, 3),
            "edges_seconds": round(self.edges_seconds, 3),
            "db_nodes": self.db_nodes,
            "db_edges": self.db_edges,
            "reconciled": self.reconciled,
            "skipped_edges": self.skipped_edges,
            "unmatched_edges": self.unmatched_edges,
        }


class Neo4jLoader:
    """Batched, constraint-first loader. All connection config via env vars."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        from neo4j import GraphDatabase

        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "")
        self.database = database or os.environ.get("NEO4J_DATABASE", "neo4j")
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _session(self):
        return self._driver.session(database=self.database)

    # -- schema ------------------------------------------------------------
    def apply_schema(self, path: Path | None = None) -> list[str]:
        """Apply every statement in schema.cypher; safe to run repeatedly."""
        text = Path(path or _SCHEMA_PATH).read_text(encoding="utf-8")
        statements = [
            s.strip()
            for s in re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE).split(";")
            if s.strip()
        ]
        applied = []
        with self._session() as s:
            for stmt in statements:
                s.run(stmt)
                applied.append(stmt.split("\n", 1)[0][:60])
        return applied

    def list_constraints(self) -> list[str]:
        with self._session() as s:
            return [r["name"] for r in s.run("SHOW CONSTRAINTS YIELD name RETURN name")]

    def index_states(self) -> dict[str, str]:
        with self._session() as s:
            return {
                r["name"]: r["state"]
                for r in s.run("SHOW INDEXES YIELD name, state RETURN name, state")
            }

    # -- helpers -----------------------------------------------------------
    def counts(self) -> dict[str, int]:
        with self._session() as s:
            n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            e = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": n, "edges": e}

    def count_where(self, predicate: str) -> int:
        with self._session() as s:
            return s.run(f"MATCH (n) WHERE {predicate} RETURN count(n) AS c").single()["c"]

    def wipe(self) -> None:
        """Drop all data (test/dev only; never called by the programme loader)."""
        with self._session() as s:
            s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")

    def _run_with_retry(self, session, cypher: str, **params) -> int:
        """Run a write batch, returning the number of rows the DB actually applied.

        The query must end in `RETURN count(...)`. Counting the *returned* rows
        rather than the submitted ones is what makes reconciliation meaningful:
        an edge whose endpoint is missing matches nothing and must not be
        reported as written (that silent gap is the failure mode the
        reconciliation gate exists to catch).
        """
        from neo4j.exceptions import TransientError

        def _work(tx):
            rec = tx.run(cypher, **params).single()
            return int(rec[0]) if rec else 0

        delay = 0.05
        for attempt in range(DEADLOCK_RETRIES):
            try:
                return session.execute_write(_work)
            except TransientError:
                if attempt == DEADLOCK_RETRIES - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return 0

    # -- load --------------------------------------------------------------
    def load(
        self,
        source: Path,
        domain: str | None = None,
        dry_run: bool = False,
        edges_source: Path | None = None,
        apply_schema: bool = True,
    ) -> LoadStats:
        """Load nodes then edges. Constraints are applied first, always."""
        stats = LoadStats()
        if apply_schema and not dry_run:
            self.apply_schema()

        ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # --- nodes: one UNWIND per batch, label taken from file_type -------
        t0 = time.perf_counter()
        with self._session() as s:
            for batch in batched(iter_nodes(source), NODE_BATCH):
                rows_by_label: dict[str, list[dict]] = {}
                for node in batch:
                    nid = node.get("id")
                    if not nid:
                        continue
                    props = sanitize_props(node)
                    props["id"] = nid
                    if domain:
                        props["domain"] = domain
                    props["ingested_at"] = ingested_at
                    label = safe_label(str(node.get("file_type", "Entity")).capitalize())
                    rows_by_label.setdefault(label, []).append(props)

                for label, rows in rows_by_label.items():
                    if dry_run:
                        stats.nodes_written += len(rows)
                        continue
                    # `Entity` carries the constraint and every index; the
                    # file_type label is additive so type filters stay cheap.
                    stats.nodes_written += self._run_with_retry(
                        s,
                        f"UNWIND $rows AS row "
                        f"MERGE (n:Entity {{id: row.id}}) "
                        f"SET n += row, n:{label} "
                        f"RETURN count(n)",
                        rows=rows,
                    )
        stats.nodes_seconds = time.perf_counter() - t0

        # --- edges: grouped by relation type, endpoints matched by id ------
        t0 = time.perf_counter()
        with self._session() as s:
            for batch in batched(iter_edges(edges_source or source), EDGE_BATCH):
                by_rel: dict[str, list[dict]] = {}
                for edge in batch:
                    src, tgt = edge.get("source"), edge.get("target")
                    if not src or not tgt:
                        stats.skipped_edges += 1
                        continue
                    props = sanitize_props(edge)
                    props.pop("source", None)
                    props.pop("target", None)
                    rel = safe_rel(str(edge.get("relation", "RELATED_TO")))
                    by_rel.setdefault(rel, []).append(
                        {"src": src, "tgt": tgt, "props": props}
                    )

                for rel, rows in by_rel.items():
                    if dry_run:
                        stats.edges_written += len(rows)
                        stats.per_relation[rel] = stats.per_relation.get(rel, 0) + len(rows)
                        continue
                    applied = self._run_with_retry(
                        s,
                        f"UNWIND $rows AS row "
                        f"MATCH (a:Entity {{id: row.src}}) "
                        f"MATCH (b:Entity {{id: row.tgt}}) "
                        f"MERGE (a)-[r:{rel}]->(b) "
                        f"SET r += row.props "
                        f"RETURN count(r)",
                        rows=rows,
                    )
                    stats.edges_written += applied
                    stats.per_relation[rel] = stats.per_relation.get(rel, 0) + applied
                    # An edge whose endpoint is absent matches nothing. Surface
                    # it instead of silently inflating the written count.
                    stats.unmatched_edges += len(rows) - applied
        stats.edges_seconds = time.perf_counter() - t0

        if not dry_run:
            db = self.counts()
            stats.db_nodes, stats.db_edges = db["nodes"], db["edges"]
            # Reconciliation asks one question: did the database apply every row
            # this run submitted? It deliberately does NOT compare against total
            # DB counts — MERGE collapses duplicate input rows, and earlier runs
            # leave rows behind, so both sides drift apart for entirely healthy
            # reasons. The real failure is an edge whose endpoint is missing
            # (unmatched) or a row with no id at all (skipped); those are the
            # silent data-loss paths, and they must be zero.
            stats.reconciled = stats.unmatched_edges == 0 and stats.skipped_edges == 0
        return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m graphify_ent.loader")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ld = sub.add_parser("load", help="load a graph.json or JSONL staging pair")
    ld.add_argument("source", type=Path)
    ld.add_argument("--edges", type=Path, default=None, help="separate edges.jsonl")
    ld.add_argument("--domain", default=None)
    ld.add_argument("--dry-run", action="store_true")
    ld.add_argument("--json", type=Path, default=None, help="write stats JSON here")

    sub.add_parser("schema", help="apply schema.cypher only")
    sub.add_parser("counts", help="print node/edge counts")

    args = ap.parse_args(argv)
    loader = Neo4jLoader()
    try:
        if args.cmd == "schema":
            for line in loader.apply_schema():
                print(f"applied: {line}")
            return 0
        if args.cmd == "counts":
            print(json.dumps(loader.counts(), indent=2))
            return 0

        stats = loader.load(
            args.source, domain=args.domain, dry_run=args.dry_run, edges_source=args.edges
        )
        payload = stats.as_dict()
        print(json.dumps(payload, indent=2))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2))
        return 0 if stats.reconciled or args.dry_run else 1
    finally:
        loader.close()


if __name__ == "__main__":
    raise SystemExit(main())
