"""Phase 5 — auth, ACL injection, audit, PII redaction.

Acceptance (execution plan §5): unauthenticated → 401; a user without a domain
claim sees ZERO nodes from that domain.
"""

from __future__ import annotations

import io
import json

import pytest

from graphify_ent.security import (
    AuditLogger,
    PIIRedactor,
    Principal,
    TokenError,
    allowed_domains,
    domain_filter_clause,
    principal_from_claims,
    verify_token,
)


class TestClaims:
    def test_extracts_domains_and_roles(self):
        p = principal_from_claims({
            "sub": "u1", "preferred_username": "alice",
            "domains": ["pilot", "fnb"], "realm_access": {"roles": ["entf-user"]},
        })
        assert p.domains == ["pilot", "fnb"] and p.username == "alice"
        assert not p.is_admin

    def test_comma_separated_domain_claim(self):
        p = principal_from_claims({"sub": "u", "domains": "pilot, fnb"})
        assert p.domains == ["pilot", "fnb"]

    def test_missing_claim_yields_no_domains(self):
        assert principal_from_claims({"sub": "u"}).domains == []

    def test_admin_role_detected(self):
        p = principal_from_claims({"sub": "u", "realm_access": {"roles": ["entf-admin"]}})
        assert p.is_admin


class TestACL:
    """The load-bearing rule: a request may narrow, never widen."""

    def test_request_is_intersected_with_entitlement(self):
        p = Principal("u", domains=["pilot"])
        assert allowed_domains(p, ["pilot", "secret"]) == ["pilot"]

    def test_no_entitlement_yields_nothing(self):
        assert allowed_domains(Principal("u", domains=[]), ["secret"]) == []

    def test_admin_may_request_anything(self):
        p = Principal("u", roles=["entf-admin"])
        assert allowed_domains(p, ["anything"]) == ["anything"]

    def test_unentitled_user_gets_an_impossible_clause_not_an_open_query(self):
        """The failure mode this guards: 'no domains' must never mean 'no filter'."""
        clause, params = domain_filter_clause(Principal("u", domains=[]), ["secret"])
        assert "false" in clause
        assert "acl_domains" not in params

    def test_entitled_user_gets_a_parameterized_filter(self):
        clause, params = domain_filter_clause(Principal("u", domains=["pilot"]), None)
        assert "$acl_domains" in clause and params["acl_domains"] == ["pilot"]

    def test_admin_without_request_is_unfiltered(self):
        clause, params = domain_filter_clause(Principal("u", roles=["entf-admin"]), None)
        assert clause == "" and params == {}

    def test_client_cannot_smuggle_a_domain_it_lacks(self):
        clause, params = domain_filter_clause(
            Principal("u", domains=["pilot"]), ["pilot", "restricted"]
        )
        assert params["acl_domains"] == ["pilot"], "widening must be impossible"


class TestTokenVerification:
    def test_empty_token_is_refused(self):
        with pytest.raises(TokenError):
            verify_token("")

    def test_unverifiable_token_is_refused_not_trusted(self, monkeypatch):
        """No verifier configured must mean refusal, never blind trust."""
        monkeypatch.delenv("OIDC_ISSUER", raising=False)
        monkeypatch.delenv("OIDC_JWKS_URL", raising=False)
        with pytest.raises(TokenError):
            verify_token("eyJhbGciOi.fake.token")


class TestAudit:
    def test_record_shape(self):
        buf = io.StringIO()
        rec = AuditLogger(buf).log_query(
            Principal("u1", username="alice", domains=["pilot"]),
            query="sauce", domains=["pilot"], node_ids=["n1", "n2"], duration_ms=12.3,
        )
        assert rec["who"] == "alice" and rec["node_count"] == 2
        logged = json.loads(buf.getvalue())
        assert logged["event"] == "query" and logged["domains"] == ["pilot"]

    def test_refusals_are_audited_too(self):
        buf = io.StringIO()
        AuditLogger(buf).log_query(Principal("u"), "kubernetes", [], [], refused=True)
        assert json.loads(buf.getvalue())["refused"] is True

    def test_node_ids_are_capped(self):
        buf = io.StringIO()
        rec = AuditLogger(buf).log_query(
            Principal("u"), "q", [], [f"n{i}" for i in range(500)]
        )
        assert len(rec["node_ids"]) == 50 and rec["node_count"] == 500


class TestPIIRedaction:
    def test_disabled_domain_passes_through(self):
        r = PIIRedactor(url="http://presidio", enabled_domains={"hr"})
        assert r.redact("Alice lives in Zurich", domain="pilot") == "Alice lives in Zurich"

    def test_enabled_without_service_fails_closed(self):
        """Never dispatch unredacted text when redaction was requested."""
        r = PIIRedactor(url="", enabled_domains={"hr"})
        with pytest.raises(RuntimeError):
            r.redact("Alice lives in Zurich", domain="hr")

    def test_unreachable_service_fails_closed(self):
        r = PIIRedactor(url="http://127.0.0.1:1", enabled_domains={"hr"})
        with pytest.raises(RuntimeError):
            r.redact("Alice", domain="hr")

    def test_enabled_for_reads_env(self, monkeypatch):
        monkeypatch.setenv("PII_REDACT_DOMAINS", "hr, legal")
        r = PIIRedactor(url="http://x")
        assert r.enabled_for("hr") and r.enabled_for("legal")
        assert not r.enabled_for("pilot")
