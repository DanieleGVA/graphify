"""Qdrant vector store — dense + sparse hybrid search.

Three collections (created idempotently on first use):

- ``chunks``   — text chunks, BGE-M3 dense (1024d) + sparse (SPLADE-style)
- ``entities`` — entity name + description, dense only
- ``facts``    — fact natural-language form, dense + temporal payload
                 (``valid_from``, ``valid_to``, ``invalidated_at``)

The temporal payload on ``facts`` is what lets the query plane (§6.1) apply
the ``as-of`` filter at the index level instead of post-filtering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

DENSE_DIM = 1024
DEFAULT_DENSE_MODEL = "BAAI/bge-m3"

CHUNKS_COLLECTION = "chunks"
ENTITIES_COLLECTION = "entities"
FACTS_COLLECTION = "facts"


@dataclass
class ChunkVector:
    id: str
    document_id: str
    chunk_index: int
    text: str
    dense: Sequence[float]
    sparse: Optional[dict[int, float]] = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class FactVector:
    id: str
    entity_id: str
    predicate: str
    object_ref: str
    polarity: int
    valid_from: datetime
    valid_to: Optional[datetime]
    invalidated_at: Optional[datetime]
    dense: Sequence[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    """Thin façade over qdrant-client. Concrete impl lives in `_qdrant` (P3)."""

    def __init__(self, url: str, *, api_key: Optional[str] = None) -> None:
        self.url = url
        self.api_key = api_key

    # --- schema ---------------------------------------------------------

    def ensure_collections(self) -> None:
        """Create the three collections if absent, with the documented schema."""
        raise NotImplementedError("VectorStore.ensure_collections — implemented in P3")

    # --- writes ---------------------------------------------------------

    def upsert_chunks(self, chunks: Iterable[ChunkVector]) -> None:
        raise NotImplementedError("VectorStore.upsert_chunks — implemented in P3")

    def upsert_facts(self, facts: Iterable[FactVector]) -> None:
        raise NotImplementedError("VectorStore.upsert_facts — implemented in P3")

    def close_fact(self, fact_id: str, *, valid_to: datetime, invalidated_by: str) -> None:
        """Mirror a graph-side fact closure into the vector payload."""
        raise NotImplementedError("VectorStore.close_fact — implemented in P3")

    # --- reads ----------------------------------------------------------

    def search_hybrid(
        self,
        query_dense: Sequence[float],
        *,
        query_sparse: Optional[dict[int, float]] = None,
        collection: str = CHUNKS_COLLECTION,
        as_of: Optional[datetime] = None,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 50,
    ) -> list[SearchResult]:
        """Hybrid (dense + sparse) search with optional ``as-of`` temporal filter.

        When ``as_of`` is provided and the collection has temporal payload
        (``facts``), results are restricted to facts active at that instant.
        """
        raise NotImplementedError("VectorStore.search_hybrid — implemented in P3")


__all__ = [
    "ChunkVector",
    "FactVector",
    "SearchResult",
    "VectorStore",
    "DENSE_DIM",
    "DEFAULT_DENSE_MODEL",
    "CHUNKS_COLLECTION",
    "ENTITIES_COLLECTION",
    "FACTS_COLLECTION",
]
