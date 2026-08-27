"""ENTERPRIPHY as a component — one console entry point for other projects.

The graph, the retriever and the verifier already exist as a library; what other
projects lacked was a doorway that does not require knowing the package layout.
This is that doorway, deliberately thin: every subcommand is a few lines over
`RetrievalService` / `Verifier`, and all configuration arrives via environment
(`NEO4J_URI`, `NEO4J_PASSWORD`, …) so the component carries no state of its own.

    graphify-ent query "Bechamel Sauce white roux" --domain pilot
    graphify-ent verify card.json --domain pilot
    graphify-ent match cards.pdf --domain canon_library     # find the reference
    graphify-ent parse cards.pdf                            # cards -> JSON, no DB
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
    from graphify_ent.verify import (CONFLICTED, CONTRADICTED, NOT_FOUND,
                                     SUPPORTED, UNPARSED, Claim, Verifier)

    verifier = Verifier(service.retriever, embed_fn=service._embed, domain=domain,
                        glossary=getattr(service.retriever, "glossary", None))
    findings = verifier.check_all([Claim(**c) for c in card["claims"]])
    counts = {v: sum(1 for f in findings if f.verdict == v)
              for v in (SUPPORTED, CONTRADICTED, NOT_FOUND, UNPARSED, CONFLICTED)}
    return {
        "card": card.get("title"),
        "claimed_reference": card.get("claimed_reference"),
        "claims": len(findings),
        "counts": counts,
        "used_pdf": False,
        "findings": [f.as_dict() for f in findings],
    }


def find_reference(service, card, registry, index, pool: int = 25) -> dict:
    """Which page of the corpus backs this card, and why.

    Retrieval finds the pages that TALK about the dish, the fingerprint ranks
    and explains those few. The inverse — ranking every recipe page by
    fingerprint — is implemented and measured NOT to work: on the export that
    declares its reference work, the dishes whose book the corpus does not hold
    scored higher than the true matches (evidence/T101). A card is a purchasing
    spec of sub-recipes; a book page teaches from raw ingredients.
    """
    import re

    from graphify_ent.recipes.ingredients import proportions
    from graphify_ent.recipes.match import RecipeQuery, confidence, explain
    from graphify_ent.recipes.techniques import techniques_in

    resolved = card.resolved(registry)
    title = re.sub(r"\([^)]*\)", " ", card.title.split(" - ")[0]).strip()
    query = RecipeQuery(title=title, resolved=resolved,
                        proportions=proportions(resolved),
                        verbs=techniques_in(card.procedure),
                        verb_seq=techniques_in(card.procedure, ordered=True))
    names = [r.canonical.replace("_", " ") for r in resolved if r.quantified][:5]
    res = service.retriever.query(" ".join([title] + names), embed_fn=service._embed,
                                  domain=index.domain,
                                  channels=("vector", "fulltext", "graph"),
                                  hops=1, result_window=pool)
    props = service.retriever.hydrate([h.node_id for h in res.hits[:pool]])
    pages = []
    for nid in [h.node_id for h in res.hits[:pool]]:
        d = props.get(nid) or {}
        if (d.get("extraction_method") or "") != "page":
            continue
        m = re.search(r"(\d+)", d.get("source_location") or "")
        cand = index.page(d.get("source_file") or "", int(m.group(1))) if m else None
        if cand is not None:
            pages.append(cand)
    ranked = index.rank(query, candidates=pages)[:3] if pages else []
    return {
        "card": card.title, "dish": title,
        "verdict": "SUPPORTED" if ranked and not res.refused else "NOT_SUPPORTED",
        "candidates": [m.as_dict() for m in ranked],
        "confidence": confidence(ranked, query, index.idf),
        "explain": explain(query, ranked[0], index.idf) if ranked else "",
    }


def cmd_parse(args) -> int:
    """Cards to JSON. No database, so another project can read its own export
    with this component's vocabulary and do whatever it likes with the rows."""
    from graphify_ent.recipes.cards import load_cards
    from graphify_ent.recipes.ingredients import Registry

    reg = Registry.load(args.registry)
    cards = load_cards(Path(args.cards))
    out = [{**c.as_dict(),
            "ingredients": [r.as_dict() for r in c.resolved(reg)],
            "proportions": c.proportions(reg)} for c in cards]
    payload = {"source": str(args.cards), "registry_version": reg.version,
               "cards": out}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{len(out)} schede -> {args.out}")
    else:
        print(text)
    return 0


def cmd_match(args) -> int:
    from graphify_ent.recipes.cards import load_cards
    from graphify_ent.recipes.ingredients import Registry
    from graphify_ent.recipes.match import CorpusIndex

    service = _service()
    reg = Registry.load(args.registry)
    index = CorpusIndex.from_graph(service.loader, args.domain, registry=reg,
                                   cache=Path(args.cache) if args.cache else None)
    index.domain = args.domain
    reports = [find_reference(service, c, reg, index, pool=args.pool)
               for c in load_cards(Path(args.cards))]
    if args.json:
        print(json.dumps({"domain": args.domain, "reports": reports},
                         indent=2, ensure_ascii=False))
    else:
        for r in reports:
            print(f"\n### {r['card'][:70]}  [{r['verdict']}]")
            print(r["explain"] or "  nessuna pagina del corpus parla di questo piatto")
    return 0


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
              f"{c['NOT_FOUND']} non trovate · {c.get('UNPARSED', 0)} illeggibili · "
              f"{c.get('CONFLICTED', 0)} in conflitto")
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

    p = sub.add_parser("parse", help="read a recipe-card export into JSON (no database)")
    p.add_argument("cards")
    p.add_argument("--registry", default=None,
                   help="ingredients.yaml alternativo (regola DOMAIN-AGNOSTIC)")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_parse)

    mt = sub.add_parser("match", help="find the corpus page that backs each card")
    mt.add_argument("cards")
    mt.add_argument("--domain", default="canon_library")
    mt.add_argument("--registry", default=None)
    mt.add_argument("--cache", default=None, help="dove tenere l'indice del corpus")
    mt.add_argument("--pool", type=int, default=25)
    mt.add_argument("--json", action="store_true")
    mt.set_defaults(fn=cmd_match)

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
