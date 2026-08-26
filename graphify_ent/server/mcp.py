"""MCP stdio transport over the same RetrievalService the REST app serves.

`RetrievalService` was written transport-independent ("FastAPI and MCP both
call these methods") and the MCP half was never built; this is it. One tool per
service method plus `verify_claim`, because the documentary-verification case
is the component's reason to exist in other projects.

Register from any project with:

    claude mcp add graphify-ent -- graphify-ent mcp

Configuration is entirely environmental (`NEO4J_URI`, `NEO4J_PASSWORD`,
`EMBED_MODEL`, …): the server owns no corpus and no credentials of its own.
All tool output is data from the graph, evidence-bound; a query the corpus
cannot support returns an explicit refusal, never a nearest guess.
"""

from __future__ import annotations

from typing import Any


def build_mcp(service=None):
    """Factory kept import-light: tests construct with a stub service and no
    running database; the real server connects lazily on first use."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("graphify-ent")
    _svc = service

    def svc():
        nonlocal _svc
        if _svc is None:
            from graphify_ent.server.app import RetrievalService
            _svc = RetrievalService()
        return _svc

    @mcp.tool()
    def query_graph(query: str, domain: str | None = None, top_k: int = 10,
                    deep: bool = False, as_of: str | None = None) -> dict[str, Any]:
        """Hybrid retrieval (vector+fulltext+graph, RRF) with explicit refusal.
        Returns evidence-bound hits with source file and location, plus any
        CONTRADICTS pairs, flagged — both sides, never silently resolved."""
        return svc().query_graph(query, domain=domain, top_k=top_k,
                                 deep=deep, as_of=as_of)

    @mcp.tool()
    def get_node(node_id: str, domain: str | None = None) -> dict[str, Any]:
        """One node with its full properties, evidence and passage included."""
        return svc().get_node(node_id, domain=domain)

    @mcp.tool()
    def get_neighbors(node_id: str, hops: int = 1, limit: int = 25,
                      domain: str | None = None) -> dict[str, Any]:
        """Typed neighborhood of a node, 1-2 hops."""
        return svc().get_neighbors(node_id, hops=hops, limit=limit, domain=domain)

    @mcp.tool()
    def shortest_path(from_id: str, to_id: str, max_hops: int = 6) -> dict[str, Any]:
        """Shortest path between two nodes, if any within max_hops."""
        return svc().shortest_path(from_id, to_id, max_hops=max_hops)

    @mcp.tool()
    def list_domains() -> dict[str, Any]:
        """Domains present in the graph with node counts."""
        return svc().list_domains()

    @mcp.tool()
    def verify_claim(subject: str, aspect: str = "", value: str = "",
                     domain: str = "pilot") -> dict[str, Any]:
        """Settle one documentary claim against the graph — never the source
        file. Verdicts: SUPPORTED / CONTRADICTED / NOT_FOUND, each carrying the
        deciding passage with book and page. NOT_FOUND is a finding, not an
        invitation to guess."""
        from graphify_ent.verify import Claim, Verifier
        s = svc()
        verifier = Verifier(s.retriever, embed_fn=s._embed, domain=domain)
        return verifier.check(Claim(subject=subject, aspect=aspect,
                                    value=value)).as_dict()

    @mcp.tool()
    def health() -> dict[str, Any]:
        """Connectivity and index state of the backing graph."""
        return svc().health()

    return mcp


def main() -> None:
    build_mcp().run()


if __name__ == "__main__":
    main()
