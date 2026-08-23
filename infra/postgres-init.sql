-- Phase 4 job-state table (replaces DynamoDB, ADR-0001).
CREATE TABLE IF NOT EXISTS chunk_state (
    chunk_id     TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    domain       TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('pending','running','done','failed','dead')),
    attempt      INT  NOT NULL DEFAULT 0,
    cost_usd     NUMERIC(12,6) NOT NULL DEFAULT 0,
    source_files TEXT[],
    staging_key  TEXT,
    error        TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunk_state_run   ON chunk_state (run_id, status);
CREATE INDEX IF NOT EXISTS chunk_state_stale ON chunk_state (status, updated_at);

-- Per-run spend ledger: the max_spend guard reads this before enqueueing.
CREATE TABLE IF NOT EXISTS run_ledger (
    run_id        TEXT PRIMARY KEY,
    domain        TEXT NOT NULL,
    max_spend_usd NUMERIC(12,6) NOT NULL,
    spent_usd     NUMERIC(12,6) NOT NULL DEFAULT 0,
    halted        BOOLEAN NOT NULL DEFAULT FALSE,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Manifest snapshots drive Phase 6.1 file-level invalidation.
CREATE TABLE IF NOT EXISTS source_manifest (
    domain      TEXT NOT NULL,
    filename    TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (domain, filename)
);
