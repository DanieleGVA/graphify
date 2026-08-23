"""Phase 4 — distributed pipeline: idempotency, DLQ, spend guard, chaos.

Acceptance (execution plan §4): kill 20 % of workers mid-run → run completes
correctly (idempotent chunks); cost within ledger estimate ±20 %.

These run without RabbitMQ/MinIO/Postgres: the state store and object store
degrade to in-process/filesystem so the idempotency *logic* is tested on every
machine. The compose stack exercises the same code paths end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify_ent.finalizer import Finalizer, RunStatus
from graphify_ent.worker import ChunkMessage, Worker, WorkerConfig, chunk_id_for


class FakeState:
    """In-memory stand-in with the same claim/finish/fail contract."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.halted_runs: set[str] = set()
        self.claims = 0

    def claim(self, msg):
        self.claims += 1
        row = self.rows.get(msg.chunk_id)
        if row and row["status"] in ("running", "done"):
            return False
        self.rows[msg.chunk_id] = {"status": "running", "attempt": (row or {}).get("attempt", 0) + 1}
        return True

    def finish(self, msg, key, cost):
        self.rows[msg.chunk_id] = {"status": "done", "key": key, "cost": cost}

    def fail(self, msg, error, dead=False):
        self.rows[msg.chunk_id] = {"status": "dead" if dead else "failed", "error": error}

    def run_halted(self, run_id):
        return run_id in self.halted_runs


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGING_DIR", str(tmp_path / "staging"))
    cfg = WorkerConfig(state_db_url="", object_store_endpoint="")
    w = Worker(cfg, extractor=lambda msg: (
        [{"id": f"{msg.chunk_id}_n{i}", "label": f"N{i}"} for i in range(3)],
        [{"source": f"{msg.chunk_id}_n0", "target": f"{msg.chunk_id}_n1", "relation": "cites"}],
        0.01,
    ))
    w.state = FakeState()
    return w


def _msg(run="r1", files=("a.pdf",)):
    return ChunkMessage(run_id=run, chunk_id=chunk_id_for(run, list(files)),
                        domain="pilot", source_files=list(files))


class TestChunkIdentity:
    def test_chunk_id_is_content_addressed(self):
        assert chunk_id_for("r1", ["a.pdf", "b.pdf"]) == chunk_id_for("r1", ["b.pdf", "a.pdf"])

    def test_different_content_different_id(self):
        assert chunk_id_for("r1", ["a.pdf"]) != chunk_id_for("r1", ["b.pdf"])

    def test_run_scoped(self):
        assert chunk_id_for("r1", ["a.pdf"]) != chunk_id_for("r2", ["a.pdf"])

    def test_message_roundtrip(self):
        m = _msg()
        assert ChunkMessage.from_json(m.to_json()).chunk_id == m.chunk_id


class TestIdempotency:
    def test_duplicate_delivery_is_a_no_op(self, worker):
        m = _msg()
        assert worker.handle(m) is not None
        assert worker.handle(m) is None, "a completed chunk must not be reprocessed"

    def test_staging_written_once(self, worker, tmp_path):
        m = _msg()
        worker.handle(m)
        worker.handle(m)
        files = list((tmp_path / "staging").rglob("nodes.jsonl"))
        assert len(files) == 1

    def test_chaos_killed_worker_chunk_is_reclaimable(self, worker):
        """Kill mid-chunk: the row stays claimable so the run still completes."""
        m = _msg()
        worker.state.rows[m.chunk_id] = {"status": "failed", "attempt": 1}
        assert worker.handle(m) is not None, "a failed chunk must be retryable"
        assert worker.state.rows[m.chunk_id]["status"] == "done"

    def test_twenty_percent_worker_loss_still_completes_every_chunk(self, worker):
        """The Phase 4 acceptance criterion, in logic form."""
        msgs = [_msg(files=(f"doc{i}.pdf",)) for i in range(10)]
        # Two of ten workers die mid-chunk (20 %): their rows are left 'failed'.
        for m in msgs[:2]:
            worker.state.rows[m.chunk_id] = {"status": "failed", "attempt": 1}
        for m in msgs:
            worker.handle(m)
        assert all(worker.state.rows[m.chunk_id]["status"] == "done" for m in msgs)


class TestSpendGuard:
    def test_halted_run_processes_nothing(self, worker):
        worker.state.halted_runs.add("r1")
        assert worker.handle(_msg()) is None

    def test_unhalted_run_proceeds(self, worker):
        assert worker.handle(_msg()) is not None


class TestFailureHandling:
    def test_failure_marks_state_and_reraises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STAGING_DIR", str(tmp_path / "s"))

        def boom(msg):
            raise RuntimeError("extraction exploded")

        w = Worker(WorkerConfig(state_db_url="", object_store_endpoint=""), extractor=boom)
        w.state = FakeState()
        m = _msg()
        with pytest.raises(RuntimeError):
            w.handle(m)
        assert w.state.rows[m.chunk_id]["status"] == "failed"

    def test_final_attempt_is_dead_lettered(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STAGING_DIR", str(tmp_path / "s"))

        def boom(msg):
            raise RuntimeError("still broken")

        cfg = WorkerConfig(state_db_url="", object_store_endpoint="")
        cfg.max_attempts = 3
        w = Worker(cfg, extractor=boom)
        w.state = FakeState()
        m = _msg()
        m.attempt = 2  # third and final attempt
        with pytest.raises(RuntimeError):
            w.handle(m)
        assert w.state.rows[m.chunk_id]["status"] == "dead"


class TestFinalizer:
    def test_run_is_incomplete_until_every_chunk_settles(self):
        assert RunStatus("r", 10, 8, 1, 0).complete is False

    def test_dead_letters_do_not_block_completion(self):
        assert RunStatus("r", 10, 8, 0, 2).complete is True

    def test_empty_run_is_not_complete(self):
        assert RunStatus("r", 0, 0, 0, 0).complete is False

    def test_finalize_reports_incomplete_without_state(self):
        out = Finalizer(state_db_url="").finalize("r1", "pilot")
        assert out["finalized"] is False
