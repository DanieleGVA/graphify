# ENTERPRIPHY — operations runbook (Phase 5)

All endpoints are environment variables (ADR-0001). Nothing here assumes a cloud provider.

## Bring the stack up

```bash
docker compose up -d              # neo4j, rabbitmq, minio, postgres, keycloak, prom/grafana/loki
docker compose up -d --scale worker=8
curl -s localhost:8100/health     # {"ok": true, "nodes": ..., "vector_index": "ONLINE"}
```

Ports are offset (7688/7475 Neo4j, 5673 RabbitMQ, 9100 MinIO, 5434 Postgres, 8081 Keycloak,
3001 Grafana) so the stack coexists with an unrelated local stack.

## Ingest a corpus

```bash
make ingest DOMAIN=pilot          # manifest diff -> chunk plan -> queue -> workers -> finalizer
```

The finalizer loads staged JSONL through the Phase 2 loader, runs the embedding job, and
reports counts. Re-running is safe: chunk ids are content-addressed and the loader MERGEs.

## Disaster recovery

The graph is a **derived artifact**; the corpus in MinIO and the semantic cache are the
durable state.

1. Restore MinIO (corpus + `entf-staging` + semantic cache).
2. Restore Postgres (`chunk_state`, `run_ledger`, `source_manifest`) — optional; without it
   the next run re-extracts everything, which is correct but pays full LLM cost.
3. `docker compose up -d neo4j && python -m graphify_ent.loader schema`
4. `make ingest DOMAIN=<d>` — a warm semantic cache makes this minutes, not hours, and ~$0
   for unchanged files.
5. `python -m graphify_ent.embed` — resumable; safe to interrupt.
6. Verify: `/health` reports `vector_index: ONLINE` and `remaining_unembedded: 0`.

## Neo4j backup

```bash
docker exec entf-neo4j neo4j-admin database dump neo4j --to-path=/backups
```
Nightly, retained 14 days. Restore with `neo4j-admin database load`. A dump is a convenience,
not the system of record — step 4 above always reconstructs the graph.

## Index rebuild

```cypher
DROP INDEX entity_embedding;
CREATE VECTOR INDEX entity_embedding FOR (n:Entity) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};
SHOW INDEXES YIELD name, state;   -- wait for ONLINE before serving traffic
```
Rebuilding does not require re-embedding: vectors live on the nodes.

## Auth

Keycloak realm `enterpriphy`; per-user `domains` claim drives the ACL. The retrieval service
derives the domain filter server-side from the token (`graphify_ent/security.py`) — a client
may narrow its request, never widen it. A principal with no `domains` claim receives a filter
that matches nothing, not an unfiltered query.

Rotate the bearer fallback (`RETRIEVAL_BEARER_TOKEN`) by restarting the retrieval service;
it is a dev-only path and should be unset once OIDC is enforced.

## Observability

- Prometheus scrapes `retrieval:8100/metrics` and `worker:9000`.
- Grafana dashboard `infra/grafana-dashboards/enterpriphy.json`: query latency p95, retrieval
  stage hit counts, refusal rate, Neo4j pool, chunks/s, $/run, cache hit-rate.
- Audit records are structured JSON on stdout → Loki. Retention: 90 days.

**Alert on the refusal rate.** A sudden drop means the support floor stopped firing and the
service may be answering without evidence — a Q1 regression, not a availability problem.

## Cost control

`MAX_SPEND_USD` per run is enforced in `run_ledger`: workers stop claiming chunks once the
run is halted. Check with:

```sql
SELECT run_id, spent_usd, max_spend_usd, halted FROM run_ledger ORDER BY started_at DESC;
```

## Re-ingest and fact invalidation

`make ingest` on a changed corpus performs the Phase 6.1 manifest diff: nodes from changed or
deleted files get `invalidated_at` and a `[:SUPERSEDED_BY]` edge to their successors.
**Nothing is deleted** — `as_of` queries continue to return the pre-change state.

Auto-resolution of clashes stays **disabled** (`dry_run=True`) until the G4 human signature.
