# Enterpriphy — Architecture (v1 proposal)

> **Naming.** The v2 evolution of this project is being rebranded to **Enterpriphy** to reflect its enterprise-scale, multi-modal, bi-temporal scope. The legacy package `graphify` (PyPI `graphifyy`, currently v0.8.x) remains in maintenance during the migration. The rebranding plan is described in [`MIGRATION_PLAN.md`](./MIGRATION_PLAN.md), Phase 0.
>
> Status: **proposal / target state**. The current shipped architecture is described in [`ARCHITECTURE.md`](./ARCHITECTURE.md). This document describes where the project is heading.

## 1. Goals and constraints

| Requirement | Target |
|---|---|
| Corpus volume | ~20 GB of mixed documents (≈ 250 k–2 M files) |
| Primary formats | PDF, DOCX, XLSX, PPTX, images (PNG/JPG/TIFF), email, code, audio/video, HTML, Markdown |
| Ingestion throughput | ≥ 50 MB/min sustained; idempotent restart |
| Query latency | < 2 s p95 for hybrid retrieval; < 8 s for multi-hop |
| Temporal reasoning | Facts carry validity intervals; new facts can **invalidate** old facts without deletion |
| Determinism | Re-ingesting the same file yields the same graph |

---

## 2. Plane overview

```
                ┌─────────────────────────────────────────────────────┐
                │                  CONTROL PLANE                       │
                │   Prefect/Temporal  •  Manifest DB  •  OpenTelemetry │
                └─────────────────────────────────────────────────────┘
                                          │
   ┌─────────────┐    ┌──────────────────┴───────────────────┐    ┌──────────────┐
   │  SOURCES    │ →  │           INGESTION PLANE             │ →  │   STORAGE    │
   │ fs / S3 /   │    │  routing → parsing → chunking →       │    │   PLANE      │
   │ gdrive /    │    │  extraction → embedding → fact-diff   │    │              │
   │ urls / git  │    │  (idempotent workers, parallel pools) │    │              │
   └─────────────┘    └───────────────────────────────────────┘    └──────┬───────┘
                                                                          │
                       ┌──────────────────────────────────────────────────┘
                       ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                          STORAGE PLANE                                    │
   │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
   │  │ Object store│ │ Vector store │  │ Graph store  │  │ Metadata DB    │ │
   │  │ MinIO / S3  │  │  Qdrant      │  │  Neo4j /     │  │  Postgres      │ │
   │  │ (blob+raw)  │  │  (dense+     │  │  Memgraph    │  │  (manifest,    │ │
   │  │             │  │   sparse)    │  │  (bi-temporal│  │   jobs, locks) │ │
   │  └────────────┘  └──────────────┘  │   property g)│  └────────────────┘ │
   │                                     └──────────────┘                     │
   └──────────────────────────────────────────────────────────────────────────┘
                       ▲
                       │
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                          QUERY PLANE                                      │
   │  MCP server  •  Hybrid retriever (BM25 + dense + graph walk) →           │
   │  Reranker (Cohere / BGE) →  Context assembler  →  Answer LLM             │
   │  +  "as-of" temporal filter (default: now)                               │
   └──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Storage plane

Four stores, each with one clear job. A single store does not scale well across 20 GB of heterogeneous assets.

| Store | Tech | Holds | Why |
|---|---|---|---|
| **Object store** | MinIO (self-host) or S3 | Original blobs + normalized renderings (PDF→page PNGs, DOCX→HTML, XLSX→sheet JSON), thumbnails, audio transcripts | De-facto standard, linear cost, native versioning |
| **Vector store** | **Qdrant** (preferred) or Weaviate | Dense embeddings (BGE-M3 multilingual, 1024-d) + sparse (SPLADE); payload with `entity_id`, `fact_id`, `valid_from/to` | Rich payload filtering, native hybrid search, mature replication/sharding |
| **Graph store** | **Neo4j 5** (already optional today) or **Memgraph** | Property graph with the bi-temporal model from §5; indexes on `(entity, valid_to)` | Cypher, APOC, GDS for community detection at scale (replaces in-memory Leiden) |
| **Metadata DB** | Postgres | File manifest (hash, mtime, status), job queue, distributed locks, idempotency keys | Transactional, reliable |

> NetworkX (current source of truth) becomes a *working representation for small subgraphs* (e.g. result of a query, viz export). The source of truth is Neo4j.

---

## 4. Ingestion plane

### 4.1 Orchestration

- **Prefect 2** (or **Temporal** if stricter retry/correctness semantics are needed): one *flow* per corpus, one *task* per file. Idempotency keyed on `content_hash` (already present in graphify as `cache.py`).
- **Queue**: Redis Streams (simple) or Kafka (multi-tenant). Start with Redis + `rq` or `dramatiq`.
- **Worker pools**: containerized, horizontally scalable. Three pools so resource contention does not collapse throughput:
  - **CPU pool** — parsing, OCR, layout analysis
  - **GPU pool** — embeddings, vision encoders
  - **LLM pool** — extraction, fact-diff (rate-limited per provider)

### 4.2 Per-file pipeline

```
file → router → parser → chunker → enricher → extractor → fact-diff → writers
```

| Stage | Component | Notes |
|---|---|---|
| **Router** | magic + extension + heuristics | Decides parser tier and worker class. Detailed in §4.3 |
| **Parser (primary)** | **LlamaParse two-tier** (see §4.3) | Cost Effective for the bulk, Agentic for complex pages |
| **Parser (fallback)** | **Docling** (IBM, OSS) | Used for air-gapped data, sensitive content that cannot leave the perimeter, or when LlamaParse rate-limits |
| **OCR (when needed)** | **Tesseract 5** + **PaddleOCR** or **Surya OCR** | Only invoked for scans when LlamaParse Agentic is not in use |
| **Image understanding** | Vision LLM (Claude Sonnet 4.6, Gemini 2.x) for captions; **CLIP** / **SigLIP** for embeddings | Caption + embedding stored as `Image` node with `DEPICTS` edges to recognized entities |
| **Excel / tables** | `openpyxl` + Docling table-former; large tables become queryable datasets (Parquet on MinIO) + a `Table` node with schema + sample rows in the graph | Bulk row values stay in Parquet; only *facts* (KPIs, targets, decisions) flow into the graph |
| **Audio / video** | `faster-whisper` (already present) + `pyannote` diarization | Speaker → person entity |
| **Code** | Tree-sitter (already present) + **SCIP** for cross-language xrefs | Already in `scip_ingest.py` |
| **Chunker** | Hierarchical (section → paragraph → sentence). Small chunks (300–500 tok) for embedding, larger (~1500 tok) for extraction | Stored as `Chunk` nodes linked to `Document` |
| **Embedder** | **BGE-M3** (multilingual, dense + sparse + ColBERT in one model) served via **TEI** or **Infinity**; alternative: OpenAI `text-embedding-3-large` | BGE-M3 is SOTA OSS and cheap to self-host |
| **Extractor (entities + facts)** | Structured-output LLM (Claude Sonnet 4.6 for quality, Haiku 4.5 for volume) with JSON-schema enforcement + few-shot per domain. Output: triples `(subject, predicate, object, polarity, sentiment, valid_from?, evidence_chunk)` | Semantic core. See §5 |
| **Fact-diff** | Compare new fact against neighborhood of existing facts → invalidate/supersede | See §5.2 |
| **Writers** | Two-phase: write `Chunk` + vectors to Qdrant, entities + facts to Neo4j, blobs to MinIO, manifest to Postgres | Outbox pattern keeps the three stores consistent |

### 4.3 Parser routing (LlamaParse two-tier + escalation classifier)

Based on [parsebench](https://parsebench.ai) results, **LlamaParse Agentic** is the only parser above 80 overall and costs ¼ of Gemini 3.1 Pro and 1⁄10 of GPT-5.5 for higher quality. **LlamaParse Cost Effective** is 3× cheaper than any competitor at comparable quality.

The router picks one of four lanes per file/page:

| Lane | When | Cost |
|---|---|---|
| `native` | `.txt`, `.md`, `.html`, `.eml`, source code | $0 |
| `docling` | Sensitive / air-gapped data, `.xlsx` (table-former is excellent and free) | $0 (compute only) |
| `cost_effective` | Default for text-heavy PDF, simple `.docx` / `.pptx` | 0.4 ¢/page |
| `agentic` | Complex layout (multi-column, dense tables, charts), scans, handwriting | 1.3 ¢/page |

#### Three-stage escalation classifier

```
                ┌──────────────────────────────────┐
   file  ───►   │  STAGE 1: File-level rules        │ ──► fixed lane (skip 2/3)
                │  (zero cost, ~µs/file)            │
                └────────────────┬─────────────────┘
                                 │ "hybrid PDF" only
                                 ▼
                ┌──────────────────────────────────┐
                │  STAGE 2: Page-level features     │ ──► soft score → lane
                │  PyMuPDF + heuristics             │
                │  (~5–20 ms/page, free)            │
                └────────────────┬─────────────────┘
                                 │ initial lane
                                 ▼
                ┌──────────────────────────────────┐
                │  STAGE 3: Quality feedback        │ ──► if Cost Effective
                │  on Cost Effective output         │     output is poor, retry
                │                                   │     the page with Agentic
                └──────────────────────────────────┘
```

**Stage 1 — file-level rules (deterministic, zero cost).** Routes ~60–75 % of real corpora without inspection.

| File / signal | Lane |
|---|---|
| `.txt`, `.md`, `.html`, `.eml`, source code | `native` |
| `.docx`, `.pptx` with < 5 embedded images | `cost_effective` |
| `.xlsx` | `docling` (table-former) |
| PDF, text-extractable on > 90 % of page area (via PyMuPDF), no dense vector graphics | `cost_effective` |
| PDF, pure scan (no text layer, full-page images) | `agentic` |
| PDF, declared tables > 5 or > 20 pages | → Stage 2 |
| PDF, hybrid (mixed text + images) | → Stage 2 |

**Stage 2 — page-level features (PyMuPDF, ~5–20 ms/page, free).**

```python
@dataclass
class PageFeatures:
    text_coverage: float          # text area / page area
    image_coverage: float
    table_count: int              # page.find_tables()
    max_table_cells: int
    column_count: int             # x-clustering of text blocks
    rotation_skew_deg: float
    has_vector_graphics: bool
    text_density: float
    is_scan_page: bool
    chart_likelihood: float
```

Hard rules (decided without scoring):

| Condition | Lane |
|---|---|
| `is_scan_page` | `agentic` |
| `column_count ≥ 3` | `agentic` |
| `max_table_cells ≥ 60` or `table_count ≥ 3` | `agentic` |
| `chart_likelihood > 0.6` | `agentic` |
| `rotation_skew_deg > 2°` | `agentic` |
| `text_coverage > 0.55` ∧ `column_count ≤ 2` ∧ `table_count ≤ 1` ∧ `image_coverage < 0.15` | `cost_effective` |

Soft score (gray zone):

```
score =  0.30 · norm(max_table_cells, 0, 60)
       + 0.25 · chart_likelihood
       + 0.20 · (column_count − 1) / 3
       + 0.15 · image_coverage
       + 0.10 · (1 − text_density_normalized)

lane = agentic if score ≥ 0.40 else cost_effective
```

Threshold `0.40` is the starting point. Re-calibrate quarterly or when the document mix shifts (§4.3.1).

**Stage 3 — quality feedback (insurance, targeted retry).** Post-hoc confidence on Cost Effective output. If below threshold, retry only the failing pages with Agentic:

| Low-quality signal | Threshold |
|---|---|
| Output chars / page area | < 30 % of document median |
| Tables expected (Stage 2) but not emitted as `\|...\|` | missing |
| Downstream JSON extraction fails / empty triples | any |
| Extracted table cells vs Stage 2 `max_table_cells` | < 50 % |
| Detected language mismatches document expected language | mismatch |

Expected retry budget: **3–8 %** of pages.

#### 4.3.1 Calibration

Bootstrap on 200–500 pages sampled from the real corpus:

1. Run **both** tiers on the sample (one-time ~$5–15).
2. Auto-label a page as "deserves Agentic" if the measured quality gap (e.g. table-cell F1 vs ground truth, output-coherence score) exceeds δ.
3. Optimize the score threshold and feature weights to maximize **F1 on the routing decision**, with constraint `expected_cost ≤ 0.6 ¢/page`.
4. Re-calibrate quarterly or when the document mix shifts.

#### 4.3.2 Expected end-to-end cost

On a typical enterprise 20 GB corpus (250–400 k pages) with distribution `65 % cost_effective / 30 % agentic / 5 % retry`:

`0.4¢ · 0.65 + 1.3¢ · 0.30 + 1.3¢ · 0.05 = 0.71 ¢/page`

→ **~$2 130 on 300 k pages**, versus $3 900 (Agentic only) or $1 200 (Cost Effective only, but at quality 71.9). Expected weighted quality: **79–81 overall**.

#### 4.3.3 Module layout

```
graphify/
  parsers/
    router.py            # entry: route_file(path) -> ParseSpec
    classifier.py        # PageFeatures + scoring
    llamaparse_client.py
    docling_fallback.py
```

Minimal API:

```python
class ParseSpec:
    lane: Literal["native", "docling", "cost_effective", "agentic"]
    page_overrides: dict[int, Literal["cost_effective", "agentic"]] | None

def route_file(path: Path) -> ParseSpec: ...
def assess_quality(page_idx: int, output: ParserOutput, features: PageFeatures) -> bool: ...
```

### 4.4 Scalability

- **CPU-bound** (parsing, OCR, Stage 2 features): N workers = N physical cores; batched per file.
- **GPU-bound** (embeddings, vision): dedicated pool; TEI / Infinity handle batching automatically.
- **LLM-bound** (extraction, fact-diff): per-provider rate limit, exponential retry, **semantic caching** on Redis keyed by `(model_id, schema_version, content_hash)`.
- **Incrementality**: the Postgres manifest stores `(file_path, content_hash, last_extracted_at, model_version, schema_version, parser_lane)`. Re-ingest only touches files whose hash changed or whose schema / model was bumped.

---

## 5. Bi-temporal fact model

This is the core requirement. The pattern is a **bi-temporal knowledge graph** as used by systems like **Graphiti** (Zep) and **Mem0**. It is worth evaluating whether to **adopt Graphiti as a library** under graphify rather than reinventing the model.

### 5.1 Fact schema

```cypher
(:Entity {id, type, name})
-[:FACT {
    predicate: "likes" | "owns" | "broke" | "bought" | ...,
    object_ref: <entity_id | literal>,
    polarity: +1 | -1,
    sentiment: float [-1..1],
    valid_from: timestamp,        // when the fact became true in the world
    valid_to: timestamp | null,   // when it ceased (null = still valid)
    recorded_at: timestamp,       // when we learned it
    invalidated_at: timestamp | null,
    invalidated_by: fact_id | null,
    confidence: float,
    evidence: [chunk_id, ...],
    source_doc: doc_id
}]->(:Entity)
```

Two temporal axes:

- **Valid time** (`valid_from` / `valid_to`): when the fact was true *in the world*.
- **Transaction time** (`recorded_at` / `invalidated_at`): when *we* knew it.

Facts are **never deleted** — only **closed** (set `valid_to` and `invalidated_at`). This gives full audit trail and enables "as-of" queries for free.

### 5.2 Fact-diff pipeline

When the extractor produces a new candidate fact `F_new = (Daniele, likes, Adidas, +1, t = 2026-05-25)`:

1. **Context retrieval**. Find existing facts about `Daniele` semantically related to `F_new` — same predicate or antonym, same object or same object category (e.g. `Shoes/Brand`). Uses a Cypher query on `(Daniele)-[:FACT]->(*)` filtered by predicate set + embedding similarity on the fact's natural-language form.
2. **Fact-to-fact relation classifier** (small deterministic LLM with strict JSON schema). For each `(F_old, F_new)` pair, label:
   - `SUPPORTS` — both remain valid
   - `SUPERSEDES` — `F_new` updates `F_old` with same polarity
   - `CONTRADICTS` — opposite polarity on same subject/object
   - `EVOLVES` — state change driven by an intermediate event
   - `UNRELATED`
3. **Apply**.
   - `CONTRADICTS` / `EVOLVES`: close `F_old` (`valid_to = F_new.valid_from`, `invalidated_at = now`, `invalidated_by = F_new.id`), insert `F_new`.
   - `SUPERSEDES`: same, with updated `confidence`.
4. **Causality**. If the context contains an event-fact (`Nike, broke, t = 2026-04-10`) between `F_old(likes Nike, +1)` and `F_new(likes Adidas, +1)`, record `(F_old)-[:INVALIDATED_BY_EVENT]->(event)`. This is what allows the explanation "the breakage caused the disappointment".

End-state for the Nike → Adidas example:

```
(Daniele)-[:FACT {pred:"likes",     obj:Nike,   polarity:+1,
                  valid_from:t1, valid_to:t2,
                  invalidated_by:F2}]->(Brand:Nike)            ← closed, not deleted

(Daniele)-[:FACT {pred:"dislikes",  obj:Nike,   polarity:-1,
                  valid_from:t2, valid_to:null,
                  caused_by:event:NikeBroke}]->(Brand:Nike)    ← active

(Daniele)-[:FACT {pred:"likes",     obj:Adidas, polarity:+1,
                  valid_from:t3, valid_to:null}]->(Brand:Adidas) ← active
```

### 5.3 "As-of" queries

Retrieval applies a temporal filter by default:

```cypher
WHERE f.valid_from <= $asOf AND (f.valid_to IS NULL OR f.valid_to > $asOf)
```

`$asOf = now()` by default. The user can ask "what did Daniele think of Nike in March 2026?" by passing a different instant. Past state stays queryable for free.

### 5.4 Why plain RAG is not enough

Plain RAG would retrieve both chunks ("I like Nikes", "I bought Adidas") and force the LLM to reconstruct the truth on every query — non-deterministic and expensive. Resolving **at ingest**, once, guarantees coherence and cheap query.

---

## 6. Query plane

### 6.1 Hybrid retriever

Three channels in parallel, fused with **Reciprocal Rank Fusion**:

1. **Dense vector** on Qdrant (BGE-M3 dense) — general semantics
2. **Sparse / BM25** (BGE-M3 sparse or SPLADE) — keywords and proper nouns
3. **Graph walk** on Neo4j from entities mentioned in the query, depth 1–2, filtered "as-of" — relationality

### 6.2 Reranker

**Cohere Rerank 3** (managed) or **BGE-reranker-v2-m3** (self-host) on top-50 → top-10.

### 6.3 Context assembler

LLM context contains:

- Top-rerank chunk snippets (with `doc:page` citations)
- Entity → active-facts subgraph (optionally with timeline: "this fact was different before 2026-04-10")
- On-demand Parquet table rows by key

### 6.4 MCP server

Extends the current `serve.py` with:

- `search(query, as_of?)`
- `entity_timeline(entity_id)` — visualize the evolution (Nike: like → dislike with cause)
- `why_invalid(fact_id)` — returns evidence + the fact that invalidated it

---

## 7. Mapping from `graphify` to `enterpriphy`

| Current module (`graphify/`) | Becomes (`enterpriphy/`) |
|---|---|
| `detect.py` | Source connector → emits events to a queue instead of returning `[Path]` |
| `extract.py` | Splits into `parsers/` (by format, §4.3) + `extractors/` (semantic, LLM); dispatch remains |
| `build.py` | Replaced by **transactional writers** (Qdrant + Neo4j + Postgres outbox) |
| `cluster.py` | Replaced by **Neo4j GDS** (distributed Leiden) — no longer in-memory |
| `analyze.py`, `report.py` | Stay, but operate on subgraphs pulled via Cypher with `as-of` filter |
| `cache.py` | Moves to Redis with TTL and versioning on `(model, schema)` |
| `serve.py` | Extended with the query tools in §6.4 |
| `semantic_cleanup.py`, `dedup.py` | Become part of fact-diff (§5.2) |
| `manifest.py` | Migrates to Postgres |
| NetworkX | Kept for small graphs (single file, query result, viz export) |

---

## 8. Minimum deployment

All self-hosted via Docker Compose / Helm:

```
- postgres                # manifest, jobs
- redis                   # queue + cache
- minio                   # blobs
- qdrant                  # vectors
- neo4j (5.x + GDS)       # graph + community detection
- tei                     # BGE-M3 embeddings, GPU
- prefect-server          # orchestration
- enterpriphy-worker      # N replicas: cpu / gpu / llm pools
- enterpriphy-api         # MCP + REST
- otel-collector + grafana + loki  # observability
```

For a realistic 20 GB corpus: one node, 16 vCPU / 64 GB RAM / 1 GPU (for embeddings) is sufficient. Neo4j and Qdrant clustering only above single-node capacity.

---

## 9. Roadmap

1. **Phase 1 — Storage split**. Introduce Postgres manifest + MinIO + Qdrant; keep current NetworkX / Neo4j for the graph. *Unlocks serious incrementality.*
2. **Phase 2 — Multi-format & parser routing**. Integrate LlamaParse two-tier with the escalation classifier (§4.3); add OCR and vision-LLM workers for images. *Unlocks the 20 GB heterogeneous corpus.*
3. **Phase 3 — Orchestration**. Prefect + Redis queue; LLM extraction on a dedicated pool with semantic caching. *Unlocks throughput.*
4. **Phase 4 — Bi-temporal facts**. Either adopt **Graphiti** under the hood, or implement the §5 schema on Neo4j + the fact-diff classifier. *Unlocks the Nike/Adidas requirement.*
5. **Phase 5 — Hybrid query**. Three-channel retriever + reranker + temporal MCP tools.
