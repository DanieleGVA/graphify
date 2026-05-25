# Enterpriphy — Migration Plan

> Companion to [`ARCHITECTURE_v2.md`](./ARCHITECTURE_v2.md). Describes how the project evolves from today's `graphify` (single-process, NetworkX in-memory, ~MB-scale corpora) to **Enterpriphy** (distributed, multi-modal, bi-temporal, ~20 GB corpora).

---

## 0. Phases at a glance

| Phase | Theme | Duration (1 senior eng) | Parallelizable with |
|---|---|---|---|
| **P0** | Rebrand `graphify` → `enterpriphy` | 1 week | — |
| **P1** | Storage split: Postgres + MinIO + Qdrant | 3–4 weeks | P2 (partial) |
| **P2** | Parser routing (LlamaParse two-tier) | 2–3 weeks | P1, P3 |
| **P3** | Orchestration & worker pools (Prefect + Redis) | 2–3 weeks | P2 |
| **P4** | Bi-temporal fact model + fact-diff | 4–5 weeks (custom) / 2–3 (Graphiti) | — (depends on P1) |
| **P5** | Hybrid query plane | 2–3 weeks | P6 |
| **P6** | Hardening, observability, launch | 2 weeks focused + cross-cutting | — |
|  | **Total** | **~16–20 dev-weeks** | ~10 calendar weeks for a team of 3 |

Sequencing assumptions:
- P0 first (everything downstream uses the new package name).
- P1 must precede P4 (the bi-temporal graph lives in Neo4j, vectors in Qdrant — both introduced by P1).
- P2 and P3 can overlap once P1 has landed the manifest.
- P5 depends on P1 (Qdrant) and P4 (temporal filter).
- P6 starts as cross-cutting from P1 and intensifies at the end.

---

## Phase 0 — Rebrand `graphify` → `enterpriphy`

**Goal.** Move the codebase to the new name without losing the existing PyPI user base or breaking Claude Code / Codex / Cursor skill integrations.

### 0.1 Workstream A — Package rename

- [ ] Create new top-level directory `enterpriphy/` as a copy of `graphify/`.
- [ ] Search-and-replace `from graphify` / `import graphify` → `from enterpriphy` / `import enterpriphy` (~150 source files, use ruff + manual review for string literals).
- [ ] Update `pyproject.toml`:
  - `name = "enterpriphy"` (new PyPI name).
  - `description` — replace marketing line.
  - `[project.scripts] enterpriphy = "enterpriphy.__main__:main"`.
  - `[tool.setuptools] packages = ["enterpriphy"]`.
  - `[tool.pyright] include = ["enterpriphy", "tests"]`.
- [ ] Update test imports under `tests/` and `pytest.ini` `testpaths`.
- [ ] Delete `graphify/` *after* P0 sign-off; until then keep both directories.

### 0.2 Workstream B — Backward-compatible shim package

- [ ] Keep publishing `graphifyy` on PyPI as a *thin shim*: it re-exports everything from `enterpriphy` and prints a `DeprecationWarning` pointing to the new package.
- [ ] Maintain the shim for **2 minor releases** then deprecate.
- [ ] Add a one-time migration script `scripts/migrate_graphify_users.py` that rewrites `import graphify` in user projects (best-effort, opt-in).

### 0.3 Workstream C — Skill files and integrations

The repo ships `skill.md`, `skill-codex.md`, `skill-opencode.md`, … (12 variants). Each declares the slash command `/graphify`. New name: `/enterpriphy` plus alias `/graphify` for one release.

- [ ] Update each `skill*.md` (name, slash command, examples).
- [ ] Update `~/.claude/skills/graphify/SKILL.md` global skill reference (out-of-repo, separate PR in user's dotfiles).
- [ ] Update `AGENTS.md`, `README.md`, `CHANGELOG.md`, `SECURITY.md`, `docs/` (logos, translations).
- [ ] Replace logo wordmarks in `docs/logo-*.svg` (design ticket).

### 0.4 Workstream D — CI / release pipeline

- [ ] New GitHub repository name `enterpriphy/enterpriphy` (or under same org); set old `graphify` repo to *archive after migration window*.
- [ ] Update `.github/workflows/*` to build/publish both packages during transition.
- [ ] Bump version to **`1.0.0-rc.0`** (semver reset signals the new product).
- [ ] PyPI: reserve `enterpriphy` name; publish first RC.

### 0.5 Acceptance criteria

- `pip install enterpriphy && enterpriphy --help` works.
- `pip install graphifyy` still works, prints a deprecation banner, and forwards calls.
- All existing tests pass under the new package name (`pytest tests/ -q`).
- Slash command `/enterpriphy` and alias `/graphify` both invoke the skill correctly in Claude Code.

---

## Phase 1 — Storage split (Postgres + MinIO + Qdrant)

**Goal.** Replace the implicit single-process state (filesystem cache + in-memory NetworkX graph) with three explicit stores. This unlocks incremental ingestion at 20 GB scale.

### 1.1 Workstream A — Postgres manifest

- [ ] Stand up Postgres in dev (`docker-compose.yml` add-on).
- [ ] Schema: `documents(id, path, content_hash, mtime, size, status, last_extracted_at, model_version, schema_version, parser_lane)`, `jobs(id, type, payload, status, attempts, …)`, `locks`.
- [ ] Port `manifest.py` and `cache.py` from JSON files to Postgres via `psycopg[binary]` + `alembic` migrations.
- [ ] Backfill script: read existing `graphify-out/.manifest.json` and load into Postgres.

### 1.2 Workstream B — MinIO object store

- [ ] Add MinIO to compose; create buckets `enterpriphy-blobs`, `enterpriphy-renderings`, `enterpriphy-thumbnails`, `enterpriphy-tables-parquet`.
- [ ] Introduce `enterpriphy/storage/blob.py` (S3 protocol via `boto3`) with `put_blob(hash, bytes) → key`, `get_blob(key)`, `presigned_url`.
- [ ] Refactor ingest stages to write to MinIO; manifest holds the blob key.

### 1.3 Workstream C — Qdrant vector store

- [ ] Add Qdrant to compose.
- [ ] Collections: `chunks` (dense 1024-d + sparse), `entities` (dense), `facts` (dense, with temporal payload).
- [ ] Add `enterpriphy/storage/vectors.py` with `upsert_chunks`, `search_hybrid(query, filters, top_k)`.
- [ ] Embedding service: deploy **TEI** with BGE-M3 model.

### 1.4 Workstream D — Outbox writer

- [ ] Introduce transactional `writers.py` implementing the **outbox pattern**: Postgres holds the source-of-truth row; a worker drains the outbox and writes to Qdrant / Neo4j; retries until ack. Prevents partial writes across stores.

### 1.5 Acceptance criteria

- Re-ingesting an unchanged file does zero LLM calls and zero vector writes (verified by metric).
- Re-ingesting after bumping `schema_version` re-runs only the affected stages.
- 1 GB corpus ingests in < 30 min on the reference machine (16 vCPU, 1 GPU).
- Crash-restart resumes from the last committed manifest row, no duplicates in Qdrant.

---

## Phase 2 — Parser routing (LlamaParse two-tier + escalation classifier)

**Goal.** Implement the parser router specified in `ARCHITECTURE_v2.md` §4.3. Drive end-to-end parser cost on the target corpus to ~0.7 ¢/page at ~80 overall quality.

### 2.1 Workstream A — Connectors

- [ ] `enterpriphy/parsers/llamaparse_client.py` — wraps both tiers (`cost_effective`, `agentic`), handles rate-limits, retries, partial-page mode.
- [ ] `enterpriphy/parsers/docling_fallback.py` — for air-gapped / sensitive flows.
- [ ] `enterpriphy/parsers/native.py` — pass-through for `.md`, `.txt`, `.html`, source code.

### 2.2 Workstream B — Classifier

- [ ] `enterpriphy/parsers/classifier.py` with `PageFeatures` dataclass and `score(features) -> float` per §4.3 of the architecture.
- [ ] PyMuPDF integration for cheap layout / table / column / scan detection.
- [ ] Stage-1 file-level rule engine.
- [ ] Stage-3 quality assessment (`assess_quality(...)`).

### 2.3 Workstream C — Router

- [ ] `enterpriphy/parsers/router.py` exposing `route_file(path) -> ParseSpec`.
- [ ] Integration with ingestion pipeline (replaces today's per-extension dispatch).

### 2.4 Workstream D — Calibration

- [ ] `scripts/calibrate_router.py`: samples 200–500 pages from the corpus, runs both tiers, computes per-page quality delta, fits the score threshold.
- [ ] Outputs a `router_config.yaml` consumed by the classifier.
- [ ] Re-run quarterly or after corpus mix shifts > 20%.

### 2.5 Workstream E — Multi-modal extensions

- [ ] Image-only files: vision LLM (Claude Sonnet 4.6 with vision) for captions; CLIP/SigLIP for embeddings → `Image` node with `DEPICTS` edges.
- [ ] Audio / video: keep `faster-whisper` (already present), add `pyannote` for speaker diarization → `Speaker` person entities.
- [ ] XLSX large tables: route to Docling table-former, materialize as Parquet on MinIO, link via `Table` node with schema.

### 2.6 Acceptance criteria

- On a 1 000-page mixed sample: weighted mean cost ≤ 0.75 ¢/page; weighted quality ≥ 78 vs ground truth on tables and content fidelity.
- Escalation retry rate ≤ 8 %.
- Air-gapped flag forces `docling` lane and triggers zero LlamaParse calls (security test).

---

## Phase 3 — Orchestration & worker pools

**Goal.** Move from a synchronous single-process pipeline to async, parallel, idempotent workers.

### 3.1 Workstream A — Queue + orchestrator

- [ ] Deploy **Prefect 2** (server + UI) and **Redis** (streams + RQ).
- [ ] Define one flow `ingest_corpus(root)` that emits one task per file.
- [ ] Idempotency key = `content_hash`; deduplication at the queue level.

### 3.2 Workstream B — Worker pools

- [ ] Three containerized worker types: **cpu** (parsing, OCR, classifier), **gpu** (embeddings, vision), **llm** (extraction, fact-diff).
- [ ] Each pool reads from its own Redis stream; autoscaling by queue depth.

### 3.3 Workstream C — Caching

- [ ] Move semantic cache from filesystem to Redis with key `(model_id, schema_version, content_hash)`.
- [ ] TTL configurable per cache class; cache hit metric exposed.

### 3.4 Acceptance criteria

- Throughput ≥ 50 MB/min sustained on 20 GB corpus, validated end-to-end.
- Killing a worker mid-task does not lose or duplicate work; the task is re-queued and reprocessed by another worker.
- Cache hit rate ≥ 80 % on a re-ingest of an unchanged corpus.

---

## Phase 4 — Bi-temporal fact model & fact-diff

**Goal.** Implement the bi-temporal property graph from `ARCHITECTURE_v2.md` §5. This is the differentiating capability — the Nike → Adidas requirement.

### 4.1 Decision point: build vs adopt Graphiti

| Path | Pros | Cons |
|---|---|---|
| **Adopt Graphiti** (Zep, OSS) under the hood | 2–3 weeks savings; mature fact-diff; community traction | Less control over schema; dependency on external roadmap; LLM choice constrained |
| **Custom on Neo4j** | Total control; tighter integration with existing `cluster.py`, `analyze.py`; bespoke predicate ontology | 4–5 weeks effort; maintenance burden |

> **Recommendation**: spike Graphiti for **1 week** on a 200-fact corpus that includes the Nike/Adidas scenario. If it handles the EVOLVES + causality edges correctly, adopt it. Otherwise build custom. *This is the most important decision in the plan.*

### 4.2 Workstream A — Schema and Neo4j setup

- [ ] Deploy Neo4j 5 with GDS plugin (compose).
- [ ] Cypher migration: `Entity`, `Fact` edge with the bi-temporal properties of §5.1.
- [ ] Indexes: `(entity_id, valid_to)`, `(predicate, valid_to)`, `(invalidated_at)`.

### 4.3 Workstream B — Extractor → triples

- [ ] LLM extractor (Claude Sonnet 4.6 + JSON-schema) producing `(subject, predicate, object, polarity, sentiment, valid_from?, evidence_chunk)`.
- [ ] Predicate ontology bootstrap (~30 starter predicates: likes, dislikes, owns, bought, broke, met, joined, …). Extensible via config.
- [ ] Entity resolution: `dedup.py` + LSH (`datasketch`, already a dependency) + LLM tiebreak.

### 4.4 Workstream C — Fact-diff engine

- [ ] `enterpriphy/facts/diff.py` implementing the four-step pipeline of §5.2.
- [ ] Relation classifier: small LLM (Haiku 4.5) with strict JSON schema, labels `SUPPORTS | SUPERSEDES | CONTRADICTS | EVOLVES | UNRELATED`.
- [ ] Closure logic: on `CONTRADICTS` / `EVOLVES` / `SUPERSEDES`, set `valid_to`, `invalidated_at`, `invalidated_by`.
- [ ] Causality: detect event-fact between two state-facts, attach `INVALIDATED_BY_EVENT` edge.

### 4.5 Workstream D — Replace in-memory NetworkX as source of truth

- [ ] `cluster.py`: replace in-memory Leiden with **Neo4j GDS Leiden** (`gds.leiden.write`); community attribute lives on the node.
- [ ] `analyze.py` / `report.py`: query subgraphs via Cypher with `as-of` filter; NetworkX kept only as a working representation for individual analyses.

### 4.6 Workstream E — Regression test suite

Critical: a battery of contradiction scenarios with ground-truth expected end-states.

- [ ] **Nike/Adidas** (the canonical case).
- [ ] **Job change** (works at X → left → works at Y).
- [ ] **Marital status** (single → married → divorced).
- [ ] **Address** (lived in A → moved to B).
- [ ] **Health condition** (had condition → recovered).
- [ ] **Preference oscillation** (likes coffee → switched to tea → back to coffee — should not collapse).
- [ ] At least 30 scenarios total, run as `pytest tests/test_facts_temporal.py`.

### 4.7 Acceptance criteria

- All 30+ scenarios pass: the active-as-of-now state matches expectations, the closed facts are present (not deleted) with correct `valid_to`, and `why_invalid(fact_id)` returns the invalidating fact and event chain.
- Query "as-of past timestamp" reconstructs the historical state correctly on at least 10 scenarios.
- Throughput: fact-diff adds ≤ 30 % overhead to extraction time per chunk.

---

## Phase 5 — Hybrid query plane

**Goal.** Replace the current MCP `serve.py` with a query layer that fuses BM25 + dense + graph walk and respects `as-of` temporal filters.

### 5.1 Workstream A — Hybrid retriever

- [ ] `enterpriphy/query/retriever.py`: three parallel channels (Qdrant dense, Qdrant sparse / SPLADE, Neo4j graph walk depth 1–2 from query entities).
- [ ] Reciprocal Rank Fusion (`k = 60`).
- [ ] Temporal filter applied at Qdrant payload level + Cypher level.

### 5.2 Workstream B — Reranker

- [ ] Pluggable: **Cohere Rerank 3** (managed) or **BGE-reranker-v2-m3** (self-host).
- [ ] Top-50 → top-10.

### 5.3 Workstream C — Context assembler

- [ ] Composes chunks + entity-active-facts subgraph + optional historical facts (when query references the past) + Parquet table rows on demand.
- [ ] Citation format: `doc:page:chunk_id`.

### 5.4 Workstream D — MCP tools

- [ ] `search(query, as_of?)`
- [ ] `entity_timeline(entity_id)` — emits the chronological sequence with invalidation events.
- [ ] `why_invalid(fact_id)` — returns evidence + invalidating fact + causal event.
- [ ] `compare(entity_id, t1, t2)` — what changed between two timestamps.

### 5.5 Acceptance criteria

- p95 query latency: < 2 s for hybrid retrieval, < 8 s for multi-hop.
- On the Nike/Adidas regression case: `search("does Daniele like Nike?")` returns the *current* state (no), `search("did Daniele ever like Nike?")` returns the *historical* yes with timeline.

---

## Phase 6 — Hardening & launch

Cross-cutting, mostly running in parallel with P1–P5, intensified for 2 focused weeks at the end.

### 6.1 Observability

- [ ] OpenTelemetry traces wired through all stages (parsing, extraction, fact-diff, query).
- [ ] Metrics: ingestion throughput, parser-tier distribution, retry rate, cache hit rate, fact-diff decision counts (`SUPPORTS|SUPERSEDES|CONTRADICTS|EVOLVES`), p50/p95/p99 query latency.
- [ ] Dashboards: Grafana boards (ingestion, query, cost). Logs to Loki.
- [ ] LLM observability: Langfuse for prompt/response tracing, token cost per stage.

### 6.2 Security

Existing `security.py` covers URL/path/label validation. New surfaces:

- [ ] Postgres / Neo4j / Qdrant / MinIO credentials via Vault or env-only (no defaults committed).
- [ ] LlamaParse / Cohere / Anthropic API keys: scoped tokens, separate dev/staging/prod.
- [ ] Tenant isolation (if multi-tenant): row-level security on Postgres, namespace per tenant in Qdrant and Neo4j.
- [ ] PII handling: configurable redactor before LLM calls; redaction map kept locally.
- [ ] Update `SECURITY.md` with the new threat model.

### 6.3 Performance & cost

- [ ] End-to-end benchmark on a 20 GB reference corpus (mixed: 50 % PDF, 20 % office, 15 % images, 10 % code, 5 % audio).
- [ ] SLO publication: throughput, query latency, $/GB-ingested, $/query.
- [ ] Capacity-planning doc.

### 6.4 Documentation

- [ ] `README.md` rewrite for Enterpriphy positioning.
- [ ] `docs/migration-from-graphify.md` for existing users.
- [ ] `docs/temporal-facts.md` with the Nike/Adidas worked example.
- [ ] API reference (Sphinx or MkDocs).

### 6.5 Testing

- [ ] Unit tests per module (continue today's policy).
- [ ] Integration tests against ephemeral Postgres / Neo4j / Qdrant / MinIO via testcontainers.
- [ ] Bi-temporal regression suite (P4.6) runs in CI on every PR.
- [ ] Performance regression: nightly job ingests a 1 GB fixed corpus and checks throughput / cost are within ±10 %.

### 6.6 Launch checklist

- [ ] Migration guide tested end-to-end with a real `graphify` user.
- [ ] All P0–P5 acceptance criteria green.
- [ ] Security review signed off.
- [ ] Bi-temporal regression suite 100 % green.
- [ ] Cut `enterpriphy 1.0.0` on PyPI; archive `graphify` repo with a redirect.

---

## Cross-cutting concerns and open decisions

| # | Decision | Owner | Deadline |
|---|---|---|---|
| D1 | Adopt **Graphiti** or build the bi-temporal layer in-house? Spike first. | Engineering lead | End of P3 |
| D2 | LLM provider mix (Anthropic only vs multi-provider): Claude Sonnet 4.6 for extraction, Haiku 4.5 for fact-diff is the proposed default. Reassess after P2 cost data. | Engineering lead | End of P2 |
| D3 | Predicate ontology: closed vocabulary (~50 predicates) vs open (LLM-emitted free-text). Closed gives consistency, open is more flexible. **Recommendation: closed, with a "proposed_predicate" review queue.** | Domain SME + Engineering | Start of P4 |
| D4 | Multi-tenancy in v1 or deferred? Affects schema design. | Product | Start of P1 |
| D5 | Managed vs self-host stance (Neo4j Aura vs Neo4j community, Qdrant Cloud vs self-host, etc.). Self-host default in this plan. | Infra | Start of P1 |
| D6 | Migration window for `graphifyy` PyPI deprecation: 2 minor releases (~6 months) proposed. | Product | End of P0 |

---

## Risk register

| Risk | Mitigation |
|---|---|
| LlamaParse rate limits at 20 GB scale | Negotiate enterprise tier early; Docling fallback proven in P2 |
| Fact-diff false positives invalidating valid facts | Conservative thresholds + audit log + `undo_invalidation` admin tool |
| Cost overrun on extraction LLM | Aggressive semantic caching (P3.3) + Haiku 4.5 for low-stakes facts + per-corpus budget gate |
| Neo4j community-edition limits at scale | Plan upgrade path to Enterprise or Aura before crossing 100 M relationships |
| Existing `graphifyy` users churn during rebrand | Shim package + clear migration guide + 6-month deprecation window |
| Bi-temporal correctness regressions | The 30-scenario test suite runs on every PR; no merges if red |
