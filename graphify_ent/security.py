"""Phase 5 — OIDC auth, per-domain ACL injection, audit log, PII redaction.

ADR-0001 substitutes Keycloak for Entra ID and Loki for CloudWatch; the OIDC
shape is unchanged, so this is a configuration swap.

The load-bearing rule: **the domain filter is derived server-side from the
token's claims and never from client input.** A client may narrow its request
to a subset of what it is entitled to; it can never widen it. `allowed_domains`
returns the intersection, and a caller with no entitlement gets an empty list,
which every query turns into "zero rows" rather than "no filter".
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AuditLogger",
    "TokenError",
    "principal_from_claims",
    "Principal",
    "PIIRedactor",
    "allowed_domains",
    "domain_filter_clause",
    "verify_token",
]


@dataclass
class Principal:
    subject: str
    username: str = ""
    domains: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        return "entf-admin" in self.roles


class TokenError(Exception):
    pass


def verify_token(token: str, jwks: dict | None = None, issuer: str | None = None) -> Principal:
    """Validate a Keycloak-issued JWT and extract the ACL claim.

    Signature verification uses the realm JWKS when `python-jose`/`PyJWT` is
    available. Without a verifier the function refuses rather than trusting the
    payload: an unverified token must never yield entitlements.
    """
    issuer = issuer or os.environ.get("OIDC_ISSUER", "")
    if not token:
        raise TokenError("missing token")

    try:
        import jwt  # PyJWT
        from jwt import PyJWKClient
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise TokenError("no JWT verifier available; refusing to trust token") from exc

    jwks_url = os.environ.get("OIDC_JWKS_URL") or (
        f"{issuer}/protocol/openid-connect/certs" if issuer else ""
    )
    if not jwks_url:
        raise TokenError("no JWKS endpoint configured")

    signing_key = PyJWKClient(jwks_url).get_signing_key_from_jwt(token).key
    claims = jwt.decode(
        token, signing_key, algorithms=["RS256"],
        audience=os.environ.get("OIDC_AUDIENCE", "account"), issuer=issuer or None,
    )
    return principal_from_claims(claims)


def principal_from_claims(claims: dict) -> Principal:
    """Map verified claims onto a Principal. Kept separate so it is testable."""
    domains = claims.get("domains") or claims.get("entf_domains") or []
    if isinstance(domains, str):
        domains = [d.strip() for d in domains.split(",") if d.strip()]
    roles = ((claims.get("realm_access") or {}).get("roles")) or []
    return Principal(
        subject=claims.get("sub", ""),
        username=claims.get("preferred_username", ""),
        domains=list(domains),
        roles=list(roles),
        raw=claims,
    )


def allowed_domains(principal: Principal, requested: str | list[str] | None = None) -> list[str]:
    """Server-side ACL resolution.

    A request may narrow, never widen. An admin sees everything; a principal
    with no domain claim sees nothing (empty list), which callers must render
    as a query that returns zero rows.
    """
    entitled = list(principal.domains)
    if principal.is_admin and not requested:
        return []  # empty + admin means "no restriction"; see domain_filter_clause
    if requested is None:
        return entitled
    req = [requested] if isinstance(requested, str) else list(requested)
    if principal.is_admin:
        return req
    return [d for d in req if d in entitled]


def domain_filter_clause(principal: Principal, requested=None, var: str = "n") -> tuple[str, dict]:
    """Return (cypher_fragment, params) enforcing the ACL server-side."""
    if principal.is_admin and requested is None:
        return "", {}
    domains = allowed_domains(principal, requested)
    if not domains:
        # No entitlement: a clause that can never match. Deliberately not "no
        # filter" — that is the failure mode this function exists to prevent.
        return f" AND {var}.domain IS NULL AND false ", {}
    return f" AND {var}.domain IN $acl_domains ", {"acl_domains": domains}


class AuditLogger:
    """Structured JSON audit records → stdout (scraped into Loki)."""

    def __init__(self, stream=None, service: str = "retrieval"):
        self.stream = stream
        self.service = service

    def log_query(
        self,
        principal: Principal,
        query: str,
        domains: list[str],
        node_ids: list[str],
        refused: bool = False,
        duration_ms: float | None = None,
    ) -> dict:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "service": self.service,
            "event": "query",
            "who": principal.username or principal.subject,
            "subject": principal.subject,
            "query": query,
            "domains": domains,
            "node_ids": node_ids[:50],
            "node_count": len(node_ids),
            "refused": refused,
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        }
        line = json.dumps(record, ensure_ascii=False)
        if self.stream is not None:
            self.stream.write(line + "\n")
        else:  # pragma: no cover - stdout path
            print(line, flush=True)
        return record


class PIIRedactor:
    """Presidio-backed redaction, opt-in per domain, applied pre-dispatch."""

    def __init__(self, url: str | None = None, enabled_domains: set[str] | None = None):
        self.url = url or os.environ.get("PRESIDIO_URL", "")
        env_domains = os.environ.get("PII_REDACT_DOMAINS", "")
        self.enabled_domains = enabled_domains or {
            d.strip() for d in env_domains.split(",") if d.strip()
        }

    def enabled_for(self, domain: str | None) -> bool:
        return bool(domain) and domain in self.enabled_domains

    def redact(self, text: str, domain: str | None = None) -> str:
        """Redact before any text leaves for the LLM endpoint.

        Fails **closed**: if redaction is enabled for the domain but the
        Presidio service cannot be reached, the text is withheld rather than
        sent unredacted.
        """
        if not self.enabled_for(domain):
            return text
        if not self.url:
            raise RuntimeError(f"PII redaction enabled for '{domain}' but PRESIDIO_URL unset")
        try:
            import urllib.request

            payload = json.dumps({"text": text, "language": "en"}).encode()
            req = urllib.request.Request(
                f"{self.url.rstrip('/')}/analyze", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                findings = json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError(f"PII redaction unavailable, refusing to dispatch: {exc}") from exc

        out = text
        for f in sorted(findings, key=lambda x: -x.get("start", 0)):
            start, end = f.get("start"), f.get("end")
            if start is None or end is None:
                continue
            out = out[:start] + f"<{f.get('entity_type', 'PII')}>" + out[end:]
        return out
