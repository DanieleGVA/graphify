"""Outbox pattern — atomic cross-store writes for Postgres + Qdrant + Neo4j.

The ingestion pipeline must keep three stores consistent: the document
manifest in Postgres, the vectors in Qdrant, the graph in Neo4j. There is
no distributed transaction across them. The outbox pattern provides
at-least-once delivery instead:

1. The producing worker writes the source-of-truth row to Postgres AND
   appends one ``outbox`` row per side-effect (e.g. ``qdrant.upsert_chunks``,
   ``neo4j.write_fact``) inside the same Postgres transaction.
2. A dedicated drainer reads the outbox in order, executes the side-effect,
   and on success marks the row delivered. On failure it retries with
   exponential backoff up to ``max_attempts``; the row stays in the outbox.
3. Side-effects are idempotent (keyed by ``content_hash`` / ``fact_id``),
   so re-delivery is safe.

This module exposes the outbox API and the drainer skeleton; the drainer
loop and Prefect deployment land in P3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Optional

OutboxStatus = Literal["pending", "delivering", "delivered", "failed"]


@dataclass
class OutboxRow:
    id: Optional[int] = None
    aggregate: str = ""        # e.g. "document", "fact"
    aggregate_id: str = ""     # e.g. content_hash, fact_id
    sink: str = ""             # e.g. "qdrant.chunks", "neo4j.fact"
    payload: dict[str, Any] = field(default_factory=dict)
    status: OutboxStatus = "pending"
    attempts: int = 0
    max_attempts: int = 8
    created_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    last_error: Optional[str] = None


SinkHandler = Callable[[OutboxRow], None]


class OutboxDrainer:
    """Polls the ``outbox`` Postgres table and dispatches rows to sink handlers.

    Skeleton only — the actual SELECT FOR UPDATE SKIP LOCKED loop, backoff,
    and metrics emission are in P3. The handler dict makes it trivial to
    register additional sinks (e.g. ``elasticsearch.docs``, ``s3.parquet``).
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._handlers: dict[str, SinkHandler] = {}

    def register(self, sink: str, handler: SinkHandler) -> None:
        self._handlers[sink] = handler

    def drain_once(self, *, batch: int = 100) -> int:
        """Process up to ``batch`` rows. Returns the number delivered."""
        raise NotImplementedError("OutboxDrainer.drain_once — implemented in P3")

    def run_forever(self, *, poll_seconds: float = 1.0) -> None:
        raise NotImplementedError("OutboxDrainer.run_forever — implemented in P3")


__all__ = ["OutboxRow", "OutboxStatus", "SinkHandler", "OutboxDrainer"]
