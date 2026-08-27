#!/usr/bin/env python3
"""Can a recipe be recognised by what it IS rather than what it is called?

The claim under test (Daniele's, 2026-08-27): recipe identity should rest on
three criteria in this order — (1) the ingredient fingerprint (ingredients AND
their proportions), (2) the procedure, (3) the title — because a menu name is
marketing ("Tangy Sweetness" is a pistachio joconde), proportions are not.

The experiment is the Tiramisù of the canon report: a modernised board formula
(gelatin, vegan cream, pâte à bombe) that a human validator matched to
Professional Baking p.495 with strength "strong". The corpus holds tiramisù in
at least three books, so the title alone cannot decide. The board card is
parsed from the report itself; the candidates are EVERY page of all 16 books
that looks like a recipe (≥3 quantity-bearing ingredient lines) — no curated
shortlist, the fingerprint has to win against the whole corpus.

Ablations, because a claim is three claims:
  * title-only ranking        (criterion 3 alone)
  * ingredients-only ranking  (criterion 1 alone)
  * combined, with the title REPLACED by a fantasy name
  * combined, with every board quantity ×3 (scale invariance)

Deterministic: no models, no graph, no network.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

REPORT = Path("../tests/CANON_VALIDATED_RECIPES_REPORT_v001.pdf")
CORPUS = Path("../canon_library")
OUT = Path("../evidence/T96/fingerprint-tiramisu.json")

#: Canonical ingredients with EN/FR/IT synonyms. Deliberately small: this is
#: the demo lexicon, not the :Concept layer it argues for.
LEXICON = {
    "mascarpone": ["mascarpone"],
    "egg": ["whole egg", "eggs whole", "egg", "eggs", "uova", "uovo", "oeuf", "œuf"],
    "egg_yolk": ["yolk", "yolks", "tuorl", "jaune"],
    "egg_white": ["egg white", "whites", "album", "blanc d"],
    "sugar": ["sugar", "zucchero", "sucre"],
    "cream": ["cream", "panna", "crème", "creme"],
    "gelatin": ["gelatin", "gelatina", "gélatine"],
    "coffee": ["coffee", "espresso", "caffè", "caffe", "café", "moka", "trablit"],
    "marsala": ["marsala"],
    "coffee_liqueur": ["kahlua", "borghetti", "liqueur coffee", "coffee liqueur"],
    "ladyfinger": ["ladyfinger", "savoiard", "cuillère", "cuillere", "sponge",
                   "biscuit"],
    "cocoa": ["cocoa", "cacao"],
    "water": ["water", "acqua", "eau"],
    "glucose": ["glucose", "glucosio", "corn syrup", "trimoline", "invert"],
    "butter": ["butter", "burro", "beurre"],
    "flour": ["flour", "farina", "farine"],
    "chocolate": ["chocolate", "cioccolato", "chocolat"],
    "milk": ["milk", "latte", "lait"],
    "vanilla": ["vanilla", "vaniglia", "vanille"],
    "rum": ["rum", "rhum"],
}

#: Procedure verbs, canonicalised across the three languages.
VERBS = {
    "WHIP": ["whip", "whisk", "beat", "montare", "monta", "sbatt", "fouett", "battre"],
    "FOLD": ["fold", "incorporar", "incorporer", "mix carefully", "mescolare delicat"],
    "SOAK": ["soak", "brush", "moisten", "imbib", "inzupp", "punch", "imbiber"],
    "BOIL": ["boil", "bollire", "bouillir", "portare a ebollizione"],
    "COOK_SUGAR": ["118°c", "120°c", "121°c", "248°f", "250°f", "cook sugar",
                   "cuocere lo zucchero", "cuire le sucre"],
    "BLOOM": ["bloom", "ammoll", "gonfiare", "ramollir", "soft gelatin"],
    "CHILL": ["chill", "freeze", "abbatt", "blast", "refroid", "raffredd"],
    "LAYER": ["layer", "spread", "strato", "étaler", "etaler", "stendere", "frame"],
    "DUST": ["dust", "spolver", "saupoudr", "sprinkle"],
    "MELT": ["melt", "fondere", "fondre", "sciogliere", "dissolve", "disolve"],
}

_QTY = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|g|mg|lb|oz|ml|cl|dl|l|pt|qt)\b", re.I)
_TO_G = {"kg": 1000.0, "g": 1.0, "mg": 0.001, "lb": 453.6, "oz": 28.35,
         "ml": 1.0, "cl": 10.0, "dl": 100.0, "l": 1000.0, "pt": 473.0, "qt": 946.0}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).lower()


def parse_ingredients(text: str) -> dict[str, float]:
    """Canonical ingredient → grams, one quantity per claim.

    First version matched a synonym anywhere and took any quantity within two
    lines. Section headers broke it twice over: "COFFEE SYRUP" absorbed the
    2,000 g of the water line beneath it, and that water line ALSO counted —
    the query became a sweet coffee liquid and its best match in 1,743 pages
    was a BBQ sauce. Now each quantity line is assigned to the nearest
    preceding ingredient line, one to one, and an ingredient line that gets no
    quantity before the next ingredient line gets nothing — which is exactly
    what a section header is.
    """
    lines = [norm(ln) for ln in text.splitlines() if ln.strip()]
    claims: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        # The LEFTMOST synonym in the line decides the canonical, because an
        # ingredient line leads with its head noun. Industrial names made
        # first-match-wins wrong twice on one card: "CREAM WHIPPING 36% UHT
        # SUGAR FREE" was classified as sugar (and the cream vanished), and
        # "COFFEE INSTANT 200G NESCAFE" leads with coffee, which is right.
        best = None
        for c, syns in LEXICON.items():
            for syn in syns:
                pos = ln.find(syn)
                if pos >= 0 and (best is None or pos < best[0]
                                 or (pos == best[0] and len(syn) > best[2])):
                    best = (pos, c, len(syn))
        if best:
            claims.append((i, best[1]))
    grams: dict[str, float] = {}
    used: set[int] = set()
    for idx, (i, canon) in enumerate(claims):
        nxt = claims[idx + 1][0] if idx + 1 < len(claims) else len(lines)
        # the quantity may share the ingredient's own line, or follow it,
        # but never past the next ingredient claim
        for j in range(i, min(i + 3, max(nxt, i + 1))):
            if j in used or j >= len(lines):
                continue
            qs = _QTY.findall(lines[j])
            if qs:
                v, u = qs[-1]
                grams[canon] = grams.get(canon, 0.0) + \
                    float(v.replace(",", ".")) * _TO_G[u.lower()]
                used.add(j)
                break
    return grams


def proportions(grams: dict[str, float]) -> dict[str, float]:
    tot = sum(grams.values())
    return {k: v / tot for k, v in grams.items()} if tot else {}


def s_ingredients(q: dict[str, float], c: dict[str, float]) -> float:
    """Histogram intersection of the proportion vectors: scale-invariant, in
    [0,1], and it rewards agreement on the RATIOS, not just co-presence —
    which is exactly criterion 1 as stated."""
    return sum(min(q.get(k, 0.0), c.get(k, 0.0)) for k in set(q) | set(c))


def parse_verbs(text: str) -> set[str]:
    body = norm(text)
    return {v for v, syns in VERBS.items() if any(s in body for s in syns)}


def s_procedure(q: set[str], c: set[str]) -> float:
    return len(q & c) / len(q | c) if q | c else 0.0


def s_title(title: str, text: str) -> float:
    toks = [t for t in re.findall(r"[a-zà-ÿ]{4,}", norm(title))]
    if not toks:
        return 0.0
    body = norm(text)
    return sum(1 for t in toks if t in body) / len(toks)


def combined(si: float, sp: float, st: float) -> float:
    return 0.60 * si + 0.25 * sp + 0.15 * st


def main() -> int:
    import fitz

    # --- the query: the board card, parsed from the report itself ---------
    with fitz.open(REPORT) as rep:
        card = next(rep[i].get_text() for i in range(rep.page_count)
                    if "Tiramis" in rep[i].get_text())
    body = card.split("Ingredients", 1)[1]
    q_grams = parse_ingredients(body)
    q_prop = proportions(q_grams)
    q_verbs = parse_verbs(card.split("Method:", 1)[1] if "Method:" in card else card)
    print("SCHEDA DI BORDO (dal report):")
    for k, v in sorted(q_prop.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<16} {q_grams[k]:>7.0f} g   {v*100:5.1f}%")
    print(f"   tecniche: {sorted(q_verbs)}")

    # --- the candidates: every recipe-looking page of all 16 books --------
    pages = []
    for f in sorted(CORPUS.iterdir()):
        if f.suffix.lower() not in (".pdf", ".epub"):
            continue
        with fitz.open(f) as doc:
            for i in range(doc.page_count):
                t = doc[i].get_text()
                g = parse_ingredients(t)
                if len(g) >= 3:
                    pages.append({"book": f.name, "page": i + 1, "text": t,
                                  "prop": proportions(g), "verbs": parse_verbs(t)})
    print(f"\ncandidati: {len(pages)} pagine-ricetta su 16 libri")

    def rank(title: str, qp: dict, qv: set, weights=(0.60, 0.25, 0.15)):
        scored = []
        for p in pages:
            si = s_ingredients(qp, p["prop"])
            sp = s_procedure(qv, p["verbs"])
            st = s_title(title, p["text"])
            scored.append((weights[0]*si + weights[1]*sp + weights[2]*st,
                           si, sp, st, p))
        scored.sort(key=lambda x: -x[0])
        return scored

    TRUE = ("Professional Baking", 489)

    def show(label, scored, n=5):
        print(f"\n=== {label} ===")
        true_rank = next((i+1 for i, (_, _, _, _, p) in enumerate(scored)
                          if TRUE[0] in p["book"] and p["page"] == TRUE[1]), None)
        for i, (s, si, sp, st, p) in enumerate(scored[:n], 1):
            mark = "<== RIFERIMENTO VERO" if (TRUE[0] in p["book"] and p["page"] == TRUE[1]) else ""
            print(f"  {i}. {s:.3f} (ingr {si:.2f} proc {sp:.2f} tit {st:.2f})  "
                  f"{p['book'][:34]:<36} p.{p['page']} {mark}")
        print(f"  --> rango del riferimento vero (Gisslen p.489): "
              f"{true_rank if true_rank else 'oltre i primi 50'}")
        return true_rank

    results = {}
    results["combinato"] = show("A. TRE CRITERI COMBINATI (0.60/0.25/0.15)",
                                rank("Tiramisù", q_prop, q_verbs))
    results["solo_titolo"] = show("B. SOLO TITOLO (criterio 3 da solo)",
                                  rank("Tiramisù", {}, set(), (0, 0, 1)))
    results["solo_ingredienti"] = show("C. SOLO IMPRONTA INGREDIENTI (criterio 1 da solo)",
                                       rank("", q_prop, set(), (1, 0, 0)))
    results["nome_fantasia"] = show("D. TITOLO DI FANTASIA «Coppa del Comandante»",
                                    rank("Coppa del Comandante", q_prop, q_verbs))
    q3 = proportions({k: v * 3 for k, v in q_grams.items()})
    results["scala_x3"] = show("E. QUANTITÀ DI BORDO ×3 (invarianza di scala)",
                               rank("Tiramisù", q3, q_verbs))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"query_grams": q_grams, "query_proportions": q_prop,
         "query_verbs": sorted(q_verbs), "candidates": len(pages),
         "true_reference": {"book": TRUE[0], "pdf_page": TRUE[1]},
         "true_rank_by_test": results,
         "weights": {"ingredients": 0.60, "procedure": 0.25, "title": 0.15}},
        indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
