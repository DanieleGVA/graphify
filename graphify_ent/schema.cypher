// ENTERPRIPHY Phase 2 — Neo4j schema. Applied idempotently BEFORE any load.
// Source: architecture doc §2.2. Vector dimension 1024 matches BGE-m3 native
// output (ADR-0001 keeps the dimension unchanged when swapping Cohere -> BGE-m3).

// --- Constraints -----------------------------------------------------------
// Uniqueness on id is what makes MERGE an index lookup instead of a label scan.
// Identity is (id, domain): two corpora may share a source book, and ids are
// namespaced per (book, slice), so `id` alone lets the second corpus MERGE onto
// the first and move its nodes into the wrong domain.
CREATE CONSTRAINT entity_id IF NOT EXISTS
  FOR (n:Entity) REQUIRE (n.id, n.domain) IS UNIQUE;

CREATE CONSTRAINT concept_id IF NOT EXISTS
  FOR (c:Concept) REQUIRE c.id IS UNIQUE;

// The composite constraint above cannot serve a lookup that knows only the id,
// and retrieval is full of them: hydration takes the fused ids, expansion takes
// the seed ids. Profiled without this index, `MATCH (n:Entity) WHERE n.id IN
// $ids` planned as NodeByLabelScan — 199,655 nodes touched to return 30, 178 ms
// on every single query, and 640 ms per graph expansion. Composite indexes are
// not usable on a prefix here, so the single-property index has to exist too.
CREATE INDEX entity_id_lookup IF NOT EXISTS FOR (n:Entity) ON (n.id);

// --- Search indexes --------------------------------------------------------
// `passage` is indexed alongside the short fields: without it lexical search
// sees only a node's label and its ~32-character quote, so an exact phrase from
// the source ("sifted flour") cannot reach the page that states it. Measured,
// that is why a documentary check returned a plausible neighbouring page
// instead of the recipe itself.
// `standard-folding` strips diacritics on both sides of the comparison.
// Without it the index is accent-exact: measured, "gruyere" matched 0 nodes
// while "gruyère" matched 14, so anyone typing a French term without its
// accents — which is what people type — retrieved nothing at all.
CREATE FULLTEXT INDEX entity_text IF NOT EXISTS
  FOR (n:Entity) ON EACH [n.label, n.text_excerpt, n.passage]
  OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard-folding'}};

CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
  FOR (n:Entity) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384,
                         `vector.similarity_function`: 'cosine'}};

// --- Filter indexes --------------------------------------------------------
CREATE INDEX entity_domain IF NOT EXISTS FOR (n:Entity) ON (n.domain);
CREATE INDEX entity_source_file IF NOT EXISTS FOR (n:Entity) ON (n.source_file);
CREATE INDEX concept_label_en IF NOT EXISTS FOR (c:Concept) ON (c.label_en);

// Bitemporal read surface (Phase 6.6): the default retrieval filter is
// `invalidated_at IS NULL AND (valid_to IS NULL OR valid_to > $now)`.
CREATE INDEX entity_invalidated_at IF NOT EXISTS FOR (n:Entity) ON (n.invalidated_at);
CREATE INDEX entity_valid_to IF NOT EXISTS FOR (n:Entity) ON (n.valid_to);
