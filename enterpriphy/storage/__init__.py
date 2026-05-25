"""Enterpriphy storage plane.

Four backends, one role each (see ARCHITECTURE_v2.md §3):

- ``manifest``  → Postgres   — files, jobs, locks, idempotency keys
- ``blob``      → MinIO / S3 — original files + normalized renderings
- ``vectors``   → Qdrant     — dense + sparse embeddings
- ``outbox``    → Postgres-backed outbox pattern — guarantees cross-store
  consistency (Qdrant + Neo4j + Postgres) without distributed transactions

Heavy dependencies (psycopg, boto3, qdrant-client, sqlalchemy, alembic)
are declared in the ``[storage]`` extra of ``pyproject.toml`` and are
imported lazily so that ``import enterpriphy.storage`` works in a stock
environment for type-checking and module discovery.
"""
from __future__ import annotations

__all__ = ["manifest", "blob", "vectors", "outbox"]
