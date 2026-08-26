#!/usr/bin/env python3
"""Which encoder finds the page that answers a cross-language question?

The claim this deposits was first produced inline and published without an
artifact — ENTF-15 flagged it as unrecomputable self-report, correctly. It is
deterministic and cheap, so it becomes a script rather than a memory.

Method: for the Q2 pairs of a chosen kind, embed the question and the PDF page
that actually answers it (the golden set's `page`, established by reading the
source, never by extraction). Report the similarity and — the figure that
decides retrieval — how often the answering page outranks the other pages in
the same set. No graph involved: this isolates the encoder.

    python tools/ablate_encoder.py --kind cross_language --only-failed evidence/T83/q2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BOOKS = {
    "Professional Chef": "The Professional Chef - The Culinary Institute of America.pdf",
    "Larousse": "Le Grand Larousse gastronomique, 6ème édition (Joël Robuchon).pdf",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("../eval/q2-golden-v1.json"))
    ap.add_argument("--corpus", type=Path, default=Path("../pilot"))
    ap.add_argument("--models", nargs="+",
                    default=["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                             "BAAI/bge-m3"])
    ap.add_argument("--kind", default="cross_language")
    ap.add_argument("--only-failed", type=Path, default=None,
                    help="a q2 run: restrict to the pairs it got wrong")
    ap.add_argument("--json", type=Path, default=Path("../evidence/T83/encoder-ab.json"))
    args = ap.parse_args()

    import fitz
    import numpy as np
    from sentence_transformers import SentenceTransformer

    pairs = [p for p in json.loads(args.golden.read_text())["pairs"]
             if p["kind"] == args.kind and p.get("page")]
    if args.only_failed:
        rows = json.loads(args.only_failed.read_text())["rows"]
        wrong = {r["id"] for r in rows if not r["mechanical"]}
        pairs = [p for p in pairs if p["id"] in wrong]
    if not pairs:
        print("nessuna coppia selezionata")
        return 1

    texts, cache = [], {}
    for p in pairs:
        name = BOOKS[p["book"]]
        if name not in cache:
            d = fitz.open(args.corpus / name)
            cache[name] = d
        page = cache[name][p["page"] - 1].get_text()
        texts.append(" ".join(page.split())[:1500])
    for d in cache.values():
        d.close()

    queries = [p["query"] for p in pairs]
    out = {"kind": args.kind, "pairs": [p["id"] for p in pairs],
           "n": len(pairs), "source": str(args.only_failed or args.golden),
           "note": "similarità query↔pagina che risponde; rango = posizione di quella "
                   "pagina fra le pagine-risposta delle altre coppie dello stesso insieme",
           "models": {}}
    for name in args.models:
        model = SentenceTransformer(name)
        qv = model.encode(queries, normalize_embeddings=True)
        dv = model.encode(texts, normalize_embeddings=True)
        sims = [float(qv[i] @ dv[i]) for i in range(len(pairs))]
        ranks = [int((qv[i] @ dv.T > sims[i]).sum()) + 1 for i in range(len(pairs))]
        out["models"][name] = {
            "similarity_mean": round(float(np.mean(sims)), 3),
            "similarity_min": round(float(np.min(sims)), 3),
            "rank1_count": sum(1 for r in ranks if r == 1),
            "per_pair": [{"id": p["id"], "similarity": round(s, 3), "rank": r}
                         for p, s, r in zip(pairs, sims, ranks)],
        }
        print(f"{name}: sim media {out['models'][name]['similarity_mean']}, "
              f"min {out['models'][name]['similarity_min']}, "
              f"prima posizione {out['models'][name]['rank1_count']}/{len(pairs)}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
