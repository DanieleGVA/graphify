"""Phase 3.1 — embedding enrichment job (BGE-m3 local).

ADR-0001 swaps Cohere embed-multilingual-v3-via-Bedrock for **BGE-m3 run
locally** (sentence-transformers). Both emit 1024-d vectors, so `schema.cypher`
is unchanged; BGE-m3 is natively multilingual (IT/EN/FR) and costs nothing per
rerun, which matters because the eval ablations re-embed repeatedly.

Resumability is the whole design: the driving query is
`MATCH (n:Entity) WHERE n.embedding IS NULL`, so a `kill -9` mid-run loses at
most one batch and the next run picks up exactly where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["EmbedStats", "Embedder", "embed_graph"]

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH = 32
EMBED_DIM = 1024


@dataclass
class EmbedStats:
    embedded: int = 0
    batches: int = 0
    seconds: float = 0.0
    model: str = DEFAULT_MODEL
    dim: int = EMBED_DIM
    cost_usd: float = 0.0  # local model: zero marginal cost, ledgered for parity
    per_batch_seconds: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "embedded": self.embedded,
            "batches": self.batches,
            "seconds": round(self.seconds, 2),
            "nodes_per_s": round(self.embedded / self.seconds, 1) if self.seconds else 0.0,
            "model": self.model,
            "dim": self.dim,
            "cost_usd": self.cost_usd,
        }


class Embedder:
    """Lazy singleton around the sentence-transformers model."""

    _model = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get("EMBED_MODEL", DEFAULT_MODEL)

    def _load(self):
        if Embedder._model is None:
            from sentence_transformers import SentenceTransformer

            Embedder._model = SentenceTransformer(self.model_name)
        return Embedder._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Normalized embeddings — cosine similarity is then a dot product."""
        model = self._load()
        vecs = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return [v.tolist() for v in vecs]


def _node_text(record) -> str:
    """What gets embedded: label + excerpt (architecture doc §3.1)."""
    label = record.get("label") or ""
    excerpt = record.get("text_excerpt") or ""
    return f"{label}\n{excerpt}".strip()[:2000]


def embed_graph(
    loader,
    batch_size: int = DEFAULT_BATCH,
    limit: int | None = None,
    model_name: str | None = None,
    ledger_path: Path | None = None,
    progress: bool = True,
) -> EmbedStats:
    """Embed every Entity lacking an embedding. Resumable by construction."""
    embedder = Embedder(model_name)
    stats = EmbedStats(model=embedder.model_name)
    t_start = time.perf_counter()

    while True:
        with loader._session() as s:
            rows = list(
                s.run(
                    "MATCH (n:Entity) WHERE n.embedding IS NULL "
                    "RETURN n.id AS id, n.label AS label, n.text_excerpt AS text_excerpt "
                    "LIMIT $k",
                    k=batch_size,
                )
            )
        if not rows:
            break

        texts = [_node_text(r) for r in rows]
        t0 = time.perf_counter()
        vectors = embedder.encode(texts)
        stats.per_batch_seconds.append(time.perf_counter() - t0)

        payload = [{"id": r["id"], "v": v} for r, v in zip(rows, vectors)]
        with loader._session() as s:
            s.execute_write(
                lambda tx: tx.run(
                    "UNWIND $rows AS row "
                    "MATCH (n:Entity {id: row.id}) "
                    "CALL db.create.setNodeVectorProperty(n, 'embedding', row.v)",
                    rows=payload,
                ).consume()
            )

        stats.embedded += len(rows)
        stats.batches += 1
        if progress and stats.batches % 10 == 0:
            print(f"  embedded {stats.embedded} nodes...", flush=True)
        if limit and stats.embedded >= limit:
            break

    stats.seconds = time.perf_counter() - t_start
    if ledger_path:
        Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ledger_path).write_text(json.dumps(stats.as_dict(), indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m graphify_ent.embed")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--ledger", type=Path, default=Path("cost-embed.json"))
    args = ap.parse_args(argv)

    from graphify_ent.loader import Neo4jLoader

    loader = Neo4jLoader()
    try:
        stats = embed_graph(
            loader, batch_size=args.batch, limit=args.limit,
            model_name=args.model, ledger_path=args.ledger,
        )
        print(json.dumps(stats.as_dict(), indent=2))
        with loader._session() as s:
            remaining = s.run(
                "MATCH (n:Entity) WHERE n.embedding IS NULL RETURN count(n) AS c"
            ).single()["c"]
            states = {
                r["name"]: r["state"]
                for r in s.run("SHOW INDEXES YIELD name, state RETURN name, state")
            }
        print(json.dumps({"remaining_unembedded": remaining,
                          "entity_embedding_state": states.get("entity_embedding")}, indent=2))
        return 0
    finally:
        loader.close()


if __name__ == "__main__":
    raise SystemExit(main())
