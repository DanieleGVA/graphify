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

from graphify_ent.loader import OVERLAP_PREFIX

__all__ = ["EmbedStats", "Embedder", "embed_graph"]

#: Chosen on measurement, not reputation: on this corpus MiniLM-L12-v2 scored
#: higher than BGE-m3 (84.9 vs 81.4 overall, 75.0 vs 66.7 cross-language),
#: encoded 39x faster and stores 384 floats instead of 1024. The speed is not a
#: nicety — the encoder was 58% of query latency, and a documentary check runs
#: one query per claim.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#: Kept reachable so a corpus already embedded the old way can be re-measured.
PREVIOUS_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH = 32
#: Vector width. Overridable because it is a property of the chosen model, not
#: of the system: BGE-m3 emits 1024, MiniLM-L12-v2 emits 384. The schema reads
#: the same value, so the two cannot silently disagree.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))

#: Characters of node text handed to the encoder (see `_node_text`).
EMBED_TEXT_CHARS = 700


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


def _device() -> str | None:
    """Where to run the encoder. `EMBED_DEVICE` wins; otherwise the platform's
    accelerator if torch reports one, and None to let sentence-transformers
    decide when torch is not importable. Measured on this corpus: the Apple GPU
    embeds the two-book graph in a fraction of the CPU time, and enrichment
    passes over 53k nodes are otherwise an hour of wall clock."""
    forced = os.environ.get("EMBED_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return None
    return None


class Embedder:
    """Lazy singleton around the sentence-transformers model."""

    _model = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get("EMBED_MODEL", DEFAULT_MODEL)

    def _load(self):
        if Embedder._model is None:
            from sentence_transformers import SentenceTransformer

            Embedder._model = SentenceTransformer(self.model_name, device=_device())
        return Embedder._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Normalized embeddings — cosine similarity is then a dot product."""
        model = self._load()
        vecs = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return [v.tolist() for v in vecs]


def _node_text(record) -> str:
    """What gets embedded: label + the widest real text the node carries.

    `passage` (the paragraph the node's quote came from) is preferred over
    `text_excerpt` when present. Measured, concept nodes hold ~32 characters of
    excerpt, so embedding the excerpt described a node by a fragment like
    "clarified butter" and the semantic channel could not tell one recipe from
    another. The passage is the same claim with its sentence attached.
    """
    label = record.get("label") or ""
    body = record.get("passage") or record.get("text_excerpt") or ""
    # Text a page borrowed from the page after it belongs to that page, not to
    # this one. It is there so a quote straddling the break can be MATCHED;
    # embedding it would blur the two pages into one vector, and would also
    # make the embedding depend on whether the overlap pass had run — the same
    # node encoding differently on two graphs that hold the same book.
    cut = body.find(OVERLAP_PREFIX)
    if cut > 0:
        body = body[:cut]
    # Bounded on purpose. A page node's passage is its whole page, and
    # embedding 3,600 characters of mixed content produces an averaged vector
    # that matches everything weakly — besides stalling the encoder. The long
    # passage exists for lexical search and for returning the evidence; the
    # vector only has to place the node in the right neighbourhood.
    return f"{label}\n{body}".strip()[:EMBED_TEXT_CHARS]


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

    # Walk the id order with a cursor instead of re-asking for "any node
    # without an embedding". That query cannot use an index — the property is
    # absent, and Neo4j does not index a missing list — so every batch rescanned
    # the whole label. Measured re-embedding 53k nodes: the rate fell from
    # ~700/min to ~420/min as the scan grew, which is the quadratic signature.
    # `id` is unique-constrained, so seeking past the cursor is an index range
    # scan. Still resumable: a restart begins at the first unembedded id.
    # The cursor is (id, domain), not id: two corpora that share a source book
    # hold the same ids, and a plain `id > cursor` skips the twin whenever a
    # batch ends exactly on a shared id. Measured after the first multi-domain
    # rebuild: 102 nodes silently left unembedded.
    cursor_id, cursor_domain = "", ""
    while True:
        with loader._session() as s:
            rows = list(
                s.run(
                    "MATCH (n:Entity) "
                    "WHERE (n.id > $cid OR (n.id = $cid AND n.domain > $cdom)) "
                    "  AND n.embedding IS NULL "
                    "RETURN n.id AS id, n.domain AS domain, n.label AS label, "
                    "n.text_excerpt AS text_excerpt, n.passage AS passage "
                    "ORDER BY n.id, n.domain LIMIT $k",
                    k=batch_size, cid=cursor_id, cdom=cursor_domain,
                )
            )
        if not rows:
            break
        cursor_id, cursor_domain = rows[-1]["id"], rows[-1]["domain"] or ""

        texts = [_node_text(r) for r in rows]
        t0 = time.perf_counter()
        vectors = embedder.encode(texts)
        stats.per_batch_seconds.append(time.perf_counter() - t0)

        # Match on (id, domain): the same id can exist in two corpora that
        # share a source book, and writing by id alone puts one corpus's vector
        # on the other's node.
        payload = [{"id": r["id"], "domain": r["domain"], "v": v}
                   for r, v in zip(rows, vectors)]
        with loader._session() as s:
            s.execute_write(
                lambda tx: tx.run(
                    "UNWIND $rows AS row "
                    "MATCH (n:Entity {id: row.id, domain: row.domain}) "
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
