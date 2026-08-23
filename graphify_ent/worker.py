"""Phase 4 — distributed extraction worker (portable stack, ADR-0001).

One worker consumes chunk messages from RabbitMQ, runs extraction for that
chunk, writes JSONL to the MinIO staging area, and marks PostgreSQL. Every
external endpoint is an env var, so the same code runs against SQS/S3/DynamoDB
by configuration alone — that is the whole point of the substitution table.

Idempotency is what makes the chaos test pass: a chunk is keyed by content, its
state row is claimed with a conditional UPDATE, and a worker killed mid-chunk
leaves the row reclaimable rather than the run broken.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ChunkMessage", "Worker", "WorkerConfig", "chunk_id_for"]


def chunk_id_for(run_id: str, paths: list[str]) -> str:
    """Content-addressed chunk id: replays are free, duplicates are no-ops."""
    payload = "|".join(sorted(paths))
    return f"{run_id}:{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


@dataclass
class ChunkMessage:
    run_id: str
    chunk_id: str
    domain: str
    source_files: list[str]
    attempt: int = 0

    @classmethod
    def from_json(cls, raw: bytes | str) -> "ChunkMessage":
        d = json.loads(raw)
        return cls(
            run_id=d["run_id"], chunk_id=d["chunk_id"], domain=d.get("domain", "default"),
            source_files=d.get("source_files", []), attempt=int(d.get("attempt", 0)),
        )

    def to_json(self) -> str:
        return json.dumps({
            "run_id": self.run_id, "chunk_id": self.chunk_id, "domain": self.domain,
            "source_files": self.source_files, "attempt": self.attempt,
        })


@dataclass
class WorkerConfig:
    queue_url: str = field(default_factory=lambda: os.environ.get("QUEUE_URL", ""))
    queue_name: str = field(default_factory=lambda: os.environ.get("QUEUE_NAME", "entf.chunks"))
    dlx_name: str = field(default_factory=lambda: os.environ.get("DLX_NAME", "entf.chunks.dlx"))
    max_attempts: int = int(os.environ.get("MAX_ATTEMPTS", "3"))
    object_store_endpoint: str = field(
        default_factory=lambda: os.environ.get("OBJECT_STORE_ENDPOINT", ""))
    staging_bucket: str = field(
        default_factory=lambda: os.environ.get("STAGING_BUCKET", "entf-staging"))
    state_db_url: str = field(default_factory=lambda: os.environ.get("STATE_DB_URL", ""))
    llm_endpoint: str = field(
        default_factory=lambda: os.environ.get("LLM_ENDPOINT", "https://api.anthropic.com"))
    max_spend_usd: float = float(os.environ.get("MAX_SPEND_USD", "50"))
    presidio_url: str = field(default_factory=lambda: os.environ.get("PRESIDIO_URL", ""))


class StateStore:
    """PostgreSQL job state (replaces DynamoDB)."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn = None

    def connect(self):
        if self._conn is None and self.dsn:
            import psycopg

            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def claim(self, msg: ChunkMessage) -> bool:
        """Conditional claim: only one worker may run a chunk at a time.

        Returns False when the chunk is already done or claimed by a live
        worker, which is what makes a duplicate delivery harmless.
        """
        conn = self.connect()
        if conn is None:
            return True  # no state store configured (single-machine mode)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunk_state (chunk_id, run_id, domain, status, attempt, source_files)
                VALUES (%s, %s, %s, 'running', 1, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                  SET status = 'running',
                      attempt = chunk_state.attempt + 1,
                      updated_at = now()
                  WHERE chunk_state.status IN ('pending', 'failed')
                     OR (chunk_state.status = 'running'
                         AND chunk_state.updated_at < now() - interval '10 minutes')
                RETURNING chunk_id
                """,
                (msg.chunk_id, msg.run_id, msg.domain, msg.source_files),
            )
            return cur.fetchone() is not None

    def finish(self, msg: ChunkMessage, staging_key: str, cost_usd: float) -> None:
        conn = self.connect()
        if conn is None:
            return
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chunk_state SET status='done', staging_key=%s, cost_usd=%s, "
                "updated_at=now() WHERE chunk_id=%s",
                (staging_key, cost_usd, msg.chunk_id),
            )
            cur.execute(
                "UPDATE run_ledger SET spent_usd = spent_usd + %s, "
                "halted = (spent_usd + %s) >= max_spend_usd WHERE run_id=%s",
                (cost_usd, cost_usd, msg.run_id),
            )

    def fail(self, msg: ChunkMessage, error: str, dead: bool = False) -> None:
        conn = self.connect()
        if conn is None:
            return
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chunk_state SET status=%s, error=%s, updated_at=now() "
                "WHERE chunk_id=%s",
                ("dead" if dead else "failed", error[:2000], msg.chunk_id),
            )

    def run_halted(self, run_id: str) -> bool:
        """The max_spend guard: stop enqueueing/processing when the run is over budget."""
        conn = self.connect()
        if conn is None:
            return False
        with conn.cursor() as cur:
            cur.execute("SELECT halted FROM run_ledger WHERE run_id=%s", (run_id,))
            row = cur.fetchone()
            return bool(row and row[0])


class ObjectStore:
    """MinIO / any S3-compatible endpoint. No cloud SDK is imported."""

    def __init__(self, cfg: WorkerConfig):
        self.cfg = cfg
        self._client = None

    def client(self):
        if self._client is None and self.cfg.object_store_endpoint:
            from minio import Minio

            endpoint = self.cfg.object_store_endpoint.replace("http://", "").replace(
                "https://", "")
            self._client = Minio(
                endpoint,
                access_key=os.environ.get("OBJECT_STORE_ACCESS_KEY", ""),
                secret_key=os.environ.get("OBJECT_STORE_SECRET_KEY", ""),
                secure=self.cfg.object_store_endpoint.startswith("https"),
            )
        return self._client

    def put_jsonl(self, key: str, rows: list[dict]) -> str:
        body = "\n".join(json.dumps(r) for r in rows).encode()
        client = self.client()
        if client is None:
            out = Path(os.environ.get("STAGING_DIR", "./staging")) / key
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(body)
            return str(out)
        import io

        if not client.bucket_exists(self.cfg.staging_bucket):
            client.make_bucket(self.cfg.staging_bucket)
        client.put_object(self.cfg.staging_bucket, key, io.BytesIO(body), len(body))
        return f"s3://{self.cfg.staging_bucket}/{key}"


class Worker:
    def __init__(self, cfg: WorkerConfig | None = None, extractor=None):
        self.cfg = cfg or WorkerConfig()
        self.state = StateStore(self.cfg.state_db_url)
        self.store = ObjectStore(self.cfg)
        self._stop = False
        #: injected so the chaos test and unit tests need no LLM
        self.extractor = extractor or self._default_extractor

    def _default_extractor(self, msg: ChunkMessage):
        from graphify_ent.pilot_extract import extract_document

        result_nodes, result_edges = [], []
        for f in msg.source_files:
            res = extract_document(Path(f), domain=msg.domain)
            result_nodes.extend(res.nodes)
            result_edges.extend(res.edges)
        return result_nodes, result_edges, 0.0

    def handle(self, msg: ChunkMessage) -> str | None:
        """Process one chunk. Returns the staging key, or None if skipped."""
        if self.state.run_halted(msg.run_id):
            return None
        if not self.state.claim(msg):
            return None  # already done or actively claimed — duplicate delivery
        try:
            nodes, edges, cost = self.extractor(msg)
            key = f"{msg.run_id}/{msg.chunk_id.split(':')[-1]}"
            self.store.put_jsonl(f"{key}/nodes.jsonl", nodes)
            self.store.put_jsonl(f"{key}/edges.jsonl", edges)
            self.state.finish(msg, key, cost)
            return key
        except Exception as exc:
            dead = msg.attempt + 1 >= self.cfg.max_attempts
            self.state.fail(msg, str(exc), dead=dead)
            raise

    def run(self) -> None:  # pragma: no cover - long-running loop
        import pika

        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        params = pika.URLParameters(self.cfg.queue_url)
        conn = pika.BlockingConnection(params)
        ch = conn.channel()
        ch.exchange_declare(self.cfg.dlx_name, exchange_type="fanout", durable=True)
        ch.queue_declare(
            self.cfg.queue_name, durable=True,
            arguments={"x-queue-type": "quorum",
                       "x-dead-letter-exchange": self.cfg.dlx_name,
                       "x-delivery-limit": self.cfg.max_attempts},
        )
        ch.basic_qos(prefetch_count=1)

        for method, _props, body in ch.consume(self.cfg.queue_name, inactivity_timeout=1):
            if self._stop:
                break
            if body is None:
                continue
            msg = ChunkMessage.from_json(body)
            try:
                self.handle(msg)
                ch.basic_ack(method.delivery_tag)
            except Exception:
                # requeue=False -> RabbitMQ routes to the DLX after the limit
                ch.basic_nack(method.delivery_tag, requeue=False)
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    Worker().run()
