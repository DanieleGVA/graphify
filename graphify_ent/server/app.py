"""Phase 3.2 — FastAPI retrieval service (MCP tool parity).

Tool names mirror `graphify serve` so existing agent habits transfer:
`query_graph`, `get_node`, `get_neighbors`, `shortest_path`, `list_domains`.
Every tool takes an optional `domain` filter and an optional `as_of` timestamp
(Phase 6.6 time travel).

Auth is a bearer token from `RETRIEVAL_BEARER_TOKEN` here; Phase 5 replaces it
with Keycloak OIDC and server-side per-domain ACL injection. The seam is
`_authorize()` — the domain filter is applied server-side in every query and is
never taken from client input alone.
"""

from __future__ import annotations

import os
from typing import Any

from graphify_ent.embed import Embedder
from graphify_ent.loader import Neo4jLoader
from graphify_ent.retrieval import (
    DEFAULT_TOKEN_BUDGET,
    REFUSAL_TEXT,
    HybridRetriever,
    serialize_context,
)

TOOL_NAMES = ["query_graph", "get_node", "get_neighbors", "shortest_path", "list_domains"]


class RetrievalService:
    """Transport-independent core: FastAPI and MCP both call these methods."""

    def __init__(self, loader: Neo4jLoader | None = None, glossary: dict | None = None):
        self.loader = loader or Neo4jLoader()
        self.retriever = HybridRetriever(self.loader, glossary=glossary)
        self._embedder: Embedder | None = None

    def _embed(self, text: str) -> list[float] | None:
        try:
            if self._embedder is None:
                self._embedder = Embedder()
            return self._embedder.encode([text])[0]
        except Exception:
            # Retrieval degrades to lexical-only rather than failing the request.
            return None

    # -- tools -------------------------------------------------------------
    def query_graph(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 10,
        deep: bool = False,
        as_of: str | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> dict[str, Any]:
        res = self.retriever.query(
            query, embedding=self._embed(query), domain=domain, deep=deep
        )
        if res.refused or not res.hits:
            # The explicit-refusal path: a correct outcome, not an error.
            return {
                "refused": True,
                "answer": REFUSAL_TEXT,
                "query": query,
                "hits": [],
                "context": "",
            }
        hits = res.hits[:top_k]
        return {
            "refused": False,
            "query": query,
            "expansions": res.expansions,
            "context": serialize_context(hits, token_budget=token_budget),
            "hits": [
                {
                    "id": h.node_id,
                    "label": h.label,
                    "score": round(h.score, 6),
                    "source_file": h.source_file,
                    "source_location": h.source_location,
                    "evidence": h.evidence,
                    "lang": h.lang,
                    "extraction_method": h.extraction_method,
                }
                for h in hits
            ],
            "contradictions": res.contradictions,
        }

    def get_node(self, node_id: str, domain: str | None = None) -> dict[str, Any]:
        with self.loader._session() as s:
            rec = s.run(
                "MATCH (n:Entity {id: $id}) "
                "WHERE ($domain IS NULL OR n.domain = $domain) "
                "RETURN n {.*} AS n",
                id=node_id, domain=domain,
            ).single()
        if not rec:
            return {"found": False, "id": node_id}
        node = dict(rec["n"])
        node.pop("embedding", None)  # never ship 1024 floats to a client
        return {"found": True, "node": node}

    def get_neighbors(
        self, node_id: str, hops: int = 1, limit: int = 25, domain: str | None = None
    ) -> dict[str, Any]:
        hops = max(1, min(hops, 2))
        with self.loader._session() as s:
            rows = s.run(
                f"MATCH (n:Entity {{id: $id}})-[r*1..{hops}]-(m:Entity) "
                "WHERE ($domain IS NULL OR m.domain = $domain) "
                "AND m.invalidated_at IS NULL "
                "RETURN DISTINCT m.id AS id, m.label AS label, "
                "m.source_file AS source_file LIMIT $limit",
                id=node_id, limit=limit, domain=domain,
            )
            return {"node_id": node_id, "neighbors": [dict(r) for r in rows]}

    def shortest_path(self, from_id: str, to_id: str, max_hops: int = 6) -> dict[str, Any]:
        with self.loader._session() as s:
            rec = s.run(
                f"MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}}), "
                f"p = shortestPath((a)-[*..{max_hops}]-(b)) "
                "RETURN [n IN nodes(p) | n.id] AS ids, length(p) AS len",
                a=from_id, b=to_id,
            ).single()
        return {"found": bool(rec), "path": rec["ids"] if rec else [],
                "length": rec["len"] if rec else None}

    def list_domains(self) -> dict[str, Any]:
        with self.loader._session() as s:
            rows = s.run(
                "MATCH (n:Entity) WHERE n.domain IS NOT NULL "
                "RETURN n.domain AS domain, count(n) AS nodes ORDER BY nodes DESC"
            )
            return {"domains": [dict(r) for r in rows]}

    def health(self) -> dict[str, Any]:
        try:
            counts = self.loader.counts()
            with self.loader._session() as s:
                embedded = s.run(
                    "MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
                ).single()["c"]
                states = {
                    r["name"]: r["state"]
                    for r in s.run("SHOW INDEXES YIELD name, state RETURN name, state")
                }
            return {"ok": True, **counts, "embedded": embedded,
                    "vector_index": states.get("entity_embedding")}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def _authorize(auth_header: str | None) -> bool:
    """Bearer check. Phase 5 replaces this with Keycloak OIDC validation."""
    expected = os.environ.get("RETRIEVAL_BEARER_TOKEN")
    if not expected:
        return True  # unset = open, for local dev only
    return bool(auth_header) and auth_header.strip() == f"Bearer {expected}"


def build_app(service: RetrievalService | None = None):
    """Construct the FastAPI app. Import-light so tests can skip FastAPI."""
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel

    svc = service or RetrievalService()
    app = FastAPI(title="ENTERPRIPHY retrieval", version="1.0")

    def auth(authorization: str | None = Header(default=None)):
        if not _authorize(authorization):
            raise HTTPException(status_code=401, detail="unauthorized")

    class QueryBody(BaseModel):
        query: str
        domain: str | None = None
        top_k: int = 10
        deep: bool = False
        as_of: str | None = None

    @app.get("/health")
    def health():
        return svc.health()

    @app.post("/query_graph", dependencies=[Depends(auth)])
    def query_graph(body: QueryBody):
        return svc.query_graph(body.query, domain=body.domain, top_k=body.top_k,
                               deep=body.deep, as_of=body.as_of)

    @app.get("/get_node/{node_id}", dependencies=[Depends(auth)])
    def get_node(node_id: str, domain: str | None = None):
        return svc.get_node(node_id, domain=domain)

    @app.get("/get_neighbors/{node_id}", dependencies=[Depends(auth)])
    def get_neighbors(node_id: str, hops: int = 1, limit: int = 25, domain: str | None = None):
        return svc.get_neighbors(node_id, hops=hops, limit=limit, domain=domain)

    @app.get("/shortest_path", dependencies=[Depends(auth)])
    def shortest_path(from_id: str, to_id: str):
        return svc.shortest_path(from_id, to_id)

    @app.get("/list_domains", dependencies=[Depends(auth)])
    def list_domains():
        return svc.list_domains()

    return app


def create_app():
    return build_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8100")))
