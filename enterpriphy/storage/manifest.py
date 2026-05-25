"""Postgres-backed file manifest and job queue.

Replaces the JSON-file manifest of legacy `graphify` (``cache.py`` +
``manifest.py``) with a transactional store. The manifest is the
source-of-truth for idempotent ingestion: every file has a row keyed by
``content_hash``; downstream stages refuse to re-process a file whose
``(content_hash, model_version, schema_version)`` already produced a row.

Schema (defined in ``alembic/versions/001_manifest.py``):

- ``documents(id, path, content_hash, mtime, size, status, last_extracted_at,
              model_version, schema_version, parser_lane, ...)``
- ``jobs(id, type, payload, status, attempts, locked_by, locked_until, ...)``
- ``locks(name, holder, acquired_at, ttl_seconds)``

This module exposes a thin functional façade — the heavy machinery
(connection pooling, retries, distributed locks) lives in
``enterpriphy.storage._pg`` (added in P3).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

ParserLane = Literal["native", "docling", "cost_effective", "agentic"]
DocStatus = Literal["pending", "parsing", "extracting", "embedded", "indexed", "failed"]

# Bump when the manifest schema changes in a backward-incompatible way.
MANIFEST_SCHEMA_VERSION = 1


@dataclass
class DocumentRow:
    """One row of the ``documents`` table."""

    id: Optional[int] = None
    path: str = ""
    content_hash: str = ""
    mtime: Optional[datetime] = None
    size: int = 0
    status: DocStatus = "pending"
    last_extracted_at: Optional[datetime] = None
    model_version: Optional[str] = None
    schema_version: int = MANIFEST_SCHEMA_VERSION
    parser_lane: Optional[ParserLane] = None
    extra: dict[str, Any] = field(default_factory=dict)


def compute_content_hash(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of file contents; the canonical idempotency key."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def needs_reprocessing(
    existing: Optional[DocumentRow],
    *,
    new_hash: str,
    model_version: str,
    schema_version: int = MANIFEST_SCHEMA_VERSION,
) -> bool:
    """Decide whether a file must re-enter the pipeline.

    Re-process if no row exists, the hash changed, the schema is bumped,
    the LLM/model version changed, or the previous run failed.
    """
    if existing is None:
        return True
    if existing.status == "failed":
        return True
    if existing.content_hash != new_hash:
        return True
    if existing.schema_version != schema_version:
        return True
    if model_version and existing.model_version != model_version:
        return True
    return False


# --- Connection façade ----------------------------------------------------

class ManifestStore:
    """Thin façade over a Postgres connection. Concrete impl lives in `_pg`.

    The skeleton accepts a DSN and exposes the minimal CRUD used by the
    ingestion plane. The connection pool, retry logic, and transactional
    boundaries land in P3 of the migration plan.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def upsert_document(self, row: DocumentRow) -> int:
        """Insert or update a document row, returning its id."""
        raise NotImplementedError("ManifestStore.upsert_document — implemented in P3")

    def get_by_hash(self, content_hash: str) -> Optional[DocumentRow]:
        raise NotImplementedError("ManifestStore.get_by_hash — implemented in P3")

    def enqueue_job(self, job_type: str, payload: dict[str, Any]) -> int:
        """Push a job onto the ``jobs`` table. Returns the job id."""
        raise NotImplementedError("ManifestStore.enqueue_job — implemented in P3")

    def claim_jobs(self, job_type: str, *, batch: int = 1, ttl_seconds: int = 300) -> list[dict[str, Any]]:
        """Atomically claim up to `batch` jobs for processing."""
        raise NotImplementedError("ManifestStore.claim_jobs — implemented in P3")

    def mark_done(self, job_id: int) -> None:
        raise NotImplementedError("ManifestStore.mark_done — implemented in P3")

    def mark_failed(self, job_id: int, error: str) -> None:
        raise NotImplementedError("ManifestStore.mark_failed — implemented in P3")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "DocumentRow",
    "ManifestStore",
    "ParserLane",
    "DocStatus",
    "MANIFEST_SCHEMA_VERSION",
    "compute_content_hash",
    "needs_reprocessing",
    "utcnow",
]
