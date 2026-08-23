"""Phase 4 — finalizer service (replaces Step Functions, ADR-0001).

Postgres-state-driven: when every chunk of a run reaches `done`, the finalizer
ingests the MinIO staging area through the Phase 2 loader, runs the Phase 3.1
embedding job, and emits a completion record. Cron-capable — it is a poll loop,
not an orchestrator DSL, so it has no cloud dependency.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Finalizer", "RunStatus"]


@dataclass
class RunStatus:
    run_id: str
    total: int
    done: int
    failed: int
    dead: int

    @property
    def complete(self) -> bool:
        # Dead-lettered chunks do not block completion; they are reported.
        return self.total > 0 and (self.done + self.dead) >= self.total

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "total": self.total, "done": self.done,
                "failed": self.failed, "dead": self.dead, "complete": self.complete}


class Finalizer:
    def __init__(self, state_db_url: str | None = None, loader=None):
        self.dsn = state_db_url or os.environ.get("STATE_DB_URL", "")
        self._conn = None
        self._loader = loader

    def _connect(self):
        if self._conn is None and self.dsn:
            import psycopg

            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def loader(self):
        if self._loader is None:
            from graphify_ent.loader import Neo4jLoader

            self._loader = Neo4jLoader()
        return self._loader

    def status(self, run_id: str) -> RunStatus:
        conn = self._connect()
        if conn is None:
            return RunStatus(run_id, 0, 0, 0, 0)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, count(*) FROM chunk_state WHERE run_id=%s GROUP BY status",
                (run_id,),
            )
            counts = dict(cur.fetchall())
        return RunStatus(
            run_id=run_id, total=sum(counts.values()),
            done=counts.get("done", 0), failed=counts.get("failed", 0),
            dead=counts.get("dead", 0),
        )

    def staging_keys(self, run_id: str) -> list[str]:
        conn = self._connect()
        if conn is None:
            return []
        with conn.cursor() as cur:
            cur.execute(
                "SELECT staging_key FROM chunk_state WHERE run_id=%s AND status='done' "
                "AND staging_key IS NOT NULL ORDER BY staging_key",
                (run_id,),
            )
            return [r[0] for r in cur.fetchall()]

    def finalize(self, run_id: str, domain: str, staging_dir: Path | None = None) -> dict:
        """Load staged JSONL → embed → report. Idempotent: MERGE-based."""
        st = self.status(run_id)
        if not st.complete:
            return {"finalized": False, **st.as_dict()}

        loader = self.loader()
        loader.apply_schema()
        totals = {"nodes": 0, "edges": 0}
        base = Path(staging_dir or os.environ.get("STAGING_DIR", "./staging"))

        for key in self.staging_keys(run_id):
            nodes_f = base / key / "nodes.jsonl"
            edges_f = base / key / "edges.jsonl"
            if not nodes_f.exists():
                continue
            stats = loader.load(nodes_f, domain=domain, edges_source=edges_f)
            totals["nodes"] += stats.nodes_written
            totals["edges"] += stats.edges_written

        from graphify_ent.embed import embed_graph

        embed_stats = embed_graph(loader, progress=False)
        return {"finalized": True, **st.as_dict(), **totals,
                "embedded": embed_stats.embedded}

    def poll(self, run_id: str, domain: str, interval: int = 30,
             max_wait: int = 3600) -> dict:  # pragma: no cover - loop
        waited = 0
        while waited < max_wait:
            out = self.finalize(run_id, domain)
            if out.get("finalized"):
                return out
            time.sleep(interval)
            waited += interval
        return {"finalized": False, "reason": "timeout"}


if __name__ == "__main__":  # pragma: no cover
    run_id = os.environ.get("RUN_ID", "")
    domain = os.environ.get("DOMAIN", "default")
    if run_id:
        print(json.dumps(Finalizer().poll(run_id, domain), indent=2))
