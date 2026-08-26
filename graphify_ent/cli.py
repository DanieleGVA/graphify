"""ENTERPRIPHY as a component — one console entry point for other projects.

The graph, the retriever and the verifier already exist as a library; what other
projects lacked was a doorway that does not require knowing the package layout.
This is that doorway, deliberately thin: every subcommand is a few lines over
`RetrievalService` / `Verifier`, and all configuration arrives via environment
(`NEO4J_URI`, `NEO4J_PASSWORD`, …) so the component carries no state of its own.

    graphify-ent query "Bechamel Sauce white roux" --domain pilot
    graphify-ent verify card.json --domain pilot
    graphify-ent health
    graphify-ent mcp          # stdio MCP server, for `claude mcp add`

Exit codes: 0 ok · 1 operational failure · 2 usage error (argparse's own).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _service():
    """Connect lazily: `--help` and usage errors must not need a database."""
    from graphify_ent.server.app import RetrievalService
    return RetrievalService()


def verify_card(service, card: dict, domain: str = "pilot") -> dict:
    """Adjudicate every claim on a card; returns the same report shape the
    T73/T74 harness writes, so downstream tooling reads one format."""
    from graphify_ent.verify import CONTRADICTED, NOT_FOUND, SUPPORTED, Claim, Verifier

    verifier = Verifier(service.retriever, embed_fn=service._embed, domain=domain)
    findings = verifier.check_all([Claim(**c) for c in card["claims"]])
    counts = {v: sum(1 for f in findings if f.verdict == v)
              for v in (SUPPORTED, CONTRADICTED, NOT_FOUND)}
    return {
        "card": card.get("title"),
        "claimed_reference": card.get("claimed_reference"),
        "claims": len(findings),
        "counts": counts,
        "used_pdf": False,
        "findings": [f.as_dict() for f in findings],
    }


def cmd_query(args) -> int:
    out = _service().query_graph(args.text, domain=args.domain, top_k=args.top)
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    if out["refused"]:
        print(out["answer"])
        return 0
    for h in out["hits"]:
        print(f"{h['score']:.3f}  {h['label'][:60]:<62} "
              f"{h['source_file'][:30]} {h['source_location'] or ''}")
    return 0


def cmd_verify(args) -> int:
    card = json.loads(Path(args.card).read_text())
    report = verify_card(_service(), card, domain=args.domain)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for f in report["findings"]:
            what = f"{f['subject']} — {f['aspect']}" if f["aspect"] else f["subject"]
            print(f"{f['verdict']:<13} {what[:70]}")
        c = report["counts"]
        print(f"\n{c['SUPPORTED']} confermate · {c['CONTRADICTED']} smentite · "
              f"{c['NOT_FOUND']} non trovate")
    return 0


def cmd_health(args) -> int:
    h = _service().health()
    print(json.dumps(h, indent=2))
    return 0 if h.get("ok") else 1


def cmd_mcp(args) -> int:
    from graphify_ent.server.mcp import main as mcp_main
    mcp_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="graphify-ent",
        description="ENTERPRIPHY component: evidence-bound retrieval and "
                    "documentary verification over the loaded knowledge graph.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query", help="hybrid retrieval with explicit refusal")
    q.add_argument("text")
    q.add_argument("--domain", default="pilot")
    q.add_argument("--top", type=int, default=10)
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_query)

    v = sub.add_parser("verify", help="adjudicate a claims card against the graph")
    v.add_argument("card")
    v.add_argument("--domain", default="pilot")
    v.add_argument("--out", help="write the JSON report here")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_verify)

    h = sub.add_parser("health", help="connectivity and index state")
    h.set_defaults(fn=cmd_health)

    m = sub.add_parser("mcp", help="serve the MCP stdio server")
    m.set_defaults(fn=cmd_mcp)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except Exception as exc:  # operational failure, not usage
        print(f"errore: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
