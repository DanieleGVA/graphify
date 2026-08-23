"""Phase 6.1–6.2 — bitemporal fact invalidation (deterministic, no LLM).

Architecture §2.3. Two axes on every node:

  transaction time  `ingested_at`, `invalidated_at`  — when the system learned/retired it
  valid time        `valid_from`, `valid_to`         — when the fact holds in the world

**Nothing is ever deleted.** Invalidation sets `invalidated_at` and links the
old node to its successor with `[:SUPERSEDED_BY]`, so `as_of` time-travel keeps
returning the pre-supersession state.

6.1 file-level (deterministic): a manifest diff says a source file changed or
    was deleted → every node from that file is invalidated, and matched to its
    re-extracted successor by dedup cluster.
6.2 version-level (heuristic): filename/metadata version markers plus a shared
    dedup cluster → the older version's nodes get `valid_to = newer.doc_date`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "InvalidationReport",
    "ManifestDiff",
    "TemporalEngine",
    "build_manifest",
    "diff_manifests",
    "file_sha256",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(root: Path, pattern: str = "*.pdf") -> dict[str, str]:
    """filename → sha256 for every matching source file."""
    return {p.name: file_sha256(p) for p in sorted(Path(root).rglob(pattern)) if p.is_file()}


@dataclass
class ManifestDiff:
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)

    @property
    def invalidating(self) -> list[str]:
        """Files whose existing nodes must be invalidated."""
        return sorted(set(self.changed) | set(self.deleted))

    def as_dict(self) -> dict:
        return {"changed": self.changed, "deleted": self.deleted, "added": self.added}


def diff_manifests(old: dict[str, str], new: dict[str, str]) -> ManifestDiff:
    return ManifestDiff(
        changed=sorted(k for k in old if k in new and old[k] != new[k]),
        deleted=sorted(k for k in old if k not in new),
        added=sorted(k for k in new if k not in old),
    )


@dataclass
class InvalidationReport:
    invalidated_nodes: int = 0
    superseded_edges: int = 0
    version_supersessions: int = 0
    files: list[str] = field(default_factory=list)
    at: str = field(default_factory=_now)

    def as_dict(self) -> dict:
        return {
            "invalidated_nodes": self.invalidated_nodes,
            "superseded_edges": self.superseded_edges,
            "version_supersessions": self.version_supersessions,
            "files": self.files,
            "at": self.at,
        }


class TemporalEngine:
    """Applies invalidation against Neo4j. Never deletes."""

    def __init__(self, loader):
        self.loader = loader

    # -- 6.1 file-level -----------------------------------------------------
    def invalidate_files(
        self, filenames: list[str], at: str | None = None, domain: str | None = None
    ) -> InvalidationReport:
        """Mark every node from these source files as no longer current."""
        rep = InvalidationReport(files=sorted(filenames), at=at or _now())
        if not filenames:
            return rep
        with self.loader._session() as s:
            rec = s.execute_write(
                lambda tx: tx.run(
                    "MATCH (n:Entity) WHERE n.source_file IN $files "
                    "AND n.invalidated_at IS NULL "
                    "AND ($domain IS NULL OR n.domain = $domain) "
                    "SET n.invalidated_at = $at "
                    "RETURN count(n) AS c",
                    files=filenames, at=rep.at, domain=domain,
                ).single()
            )
            rep.invalidated_nodes = rec["c"] if rec else 0
        return rep

    def link_successors(
        self, pairs: list[tuple[str, str]], rationale: str = "manifest-diff"
    ) -> int:
        """Link old → new nodes with [:SUPERSEDED_BY], carrying a policy trail."""
        if not pairs:
            return 0
        rows = [{"old": o, "new": n} for o, n in pairs]
        with self.loader._session() as s:
            rec = s.execute_write(
                lambda tx: tx.run(
                    "UNWIND $rows AS row "
                    "MATCH (o:Entity {id: row.old}) MATCH (n:Entity {id: row.new}) "
                    "MERGE (o)-[r:SUPERSEDED_BY]->(n) "
                    "SET r.detected_at = $at, r.rationale = $rationale, r.policy = 'file-level' "
                    "RETURN count(r) AS c",
                    rows=rows, at=_now(), rationale=rationale,
                ).single()
            )
            return rec["c"] if rec else 0

    def reingest_invalidate(
        self,
        old_manifest: dict[str, str],
        new_manifest: dict[str, str],
        domain: str | None = None,
    ) -> InvalidationReport:
        """The 6.1 entry point: diff manifests, invalidate what changed."""
        diff = diff_manifests(old_manifest, new_manifest)
        return self.invalidate_files(diff.invalidating, domain=domain)

    # -- 6.2 version-level --------------------------------------------------
    def apply_version_supersession(
        self, domain: str | None = None, dedup_key: str = "label_en"
    ) -> InvalidationReport:
        """Older versions of the same document lose to newer ones.

        Matching is deliberately conservative: same normalized dedup key *and*
        both documents carry a numeric version marker, so `policy_v1` /
        `policy_v2` supersedes but two unrelated documents never do.
        """
        rep = InvalidationReport()
        with self.loader._session() as s:
            rows = list(
                s.run(
                    f"MATCH (older:Entity), (newer:Entity) "
                    f"WHERE older.{dedup_key} IS NOT NULL "
                    f"AND older.{dedup_key} = newer.{dedup_key} "
                    f"AND older.source_file <> newer.source_file "
                    f"AND older.version IS NOT NULL AND newer.version IS NOT NULL "
                    f"AND older.version < newer.version "
                    f"AND older.invalidated_at IS NULL "
                    f"AND ($domain IS NULL OR older.domain = $domain) "
                    f"RETURN older.id AS old_id, newer.id AS new_id, "
                    f"newer.doc_date AS new_date",
                    domain=domain,
                )
            )
            if not rows:
                return rep
            payload = [
                {"old": r["old_id"], "new": r["new_id"], "valid_to": r["new_date"]}
                for r in rows
            ]
            rec = s.execute_write(
                lambda tx: tx.run(
                    "UNWIND $rows AS row "
                    "MATCH (o:Entity {id: row.old}) MATCH (n:Entity {id: row.new}) "
                    "SET o.valid_to = coalesce(row.valid_to, o.valid_to) "
                    "MERGE (o)-[r:SUPERSEDED_BY]->(n) "
                    "SET r.detected_at = $at, r.policy = 'version-level', "
                    "    r.rationale = 'newer numeric version of the same document' "
                    "RETURN count(r) AS c",
                    rows=payload, at=_now(),
                ).single()
            )
            rep.version_supersessions = rec["c"] if rec else 0
            rep.superseded_edges = rep.version_supersessions
        return rep

    # -- read surface -------------------------------------------------------
    def valid_nodes(self, as_of: str | None = None, domain: str | None = None) -> int:
        """Count of currently-valid nodes, or of nodes valid at `as_of`."""
        with self.loader._session() as s:
            if as_of:
                rec = s.run(
                    "MATCH (n:Entity) WHERE ($domain IS NULL OR n.domain = $domain) "
                    "AND (n.invalidated_at IS NULL OR n.invalidated_at > $as_of) "
                    "AND (n.valid_to IS NULL OR n.valid_to > $as_of) "
                    "RETURN count(n) AS c",
                    as_of=as_of, domain=domain,
                ).single()
            else:
                rec = s.run(
                    "MATCH (n:Entity) WHERE ($domain IS NULL OR n.domain = $domain) "
                    "AND n.invalidated_at IS NULL "
                    "RETURN count(n) AS c",
                    domain=domain,
                ).single()
            return rec["c"] if rec else 0

    def save_manifest(self, manifest: dict[str, str], path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    @staticmethod
    def load_manifest(path: Path) -> dict[str, str]:
        p = Path(path)
        return json.loads(p.read_text()) if p.exists() else {}
