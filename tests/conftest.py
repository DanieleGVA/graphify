"""Test-suite wiring.

**Database isolation.** Several integration tests wipe the database or load
20k-node benchmark graphs. Pointing those at the same Neo4j that holds a real
corpus destroys it — that happened twice during development, each time costing a
full re-embed. So the suite redirects to a dedicated test instance:

    NEO4J_TEST_URI   (preferred) — an instance the tests may freely destroy
    NEO4J_URI        (fallback)  — used only if no test instance is configured

Start one with:

    docker run -d --name entf-neo4j-test -p 7689:7687 \
        -e NEO4J_AUTH=neo4j/enterpriphy neo4j:5
    export NEO4J_TEST_URI=bolt://localhost:7689
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    test_uri = os.environ.get("NEO4J_TEST_URI")
    if test_uri:
        os.environ["NEO4J_URI"] = test_uri
        return

    if os.environ.get("NEO4J_URI") and not os.environ.get("ENTF_ALLOW_DESTRUCTIVE_TESTS"):
        # Refuse to run destructive integration tests against an unknown
        # database. Skipping is the safe default; opt in explicitly.
        os.environ.pop("NEO4J_URI", None)
        config._entf_neo4j_note = (
            "NEO4J_URI was set but NEO4J_TEST_URI was not: Neo4j integration tests were "
            "skipped rather than risk wiping a live corpus. Set NEO4J_TEST_URI to a "
            "throwaway instance, or ENTF_ALLOW_DESTRUCTIVE_TESTS=1 to override."
        )


def pytest_report_header(config):
    note = getattr(config, "_entf_neo4j_note", None)
    return [note] if note else []
