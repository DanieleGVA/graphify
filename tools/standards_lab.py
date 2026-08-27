#!/usr/bin/env python3
"""Validate the procedure and gravimetry standards BEFORE implementing them.

Four measurements, per docs/proposta-standard-procedure-gravimetria.md §3:

  1. Gravimetry against the books' own truth. Anglo-American books write dual
     units — "¾ cup (180 mL)", "6 oz (170 g)" — and every such line is a free
     ground-truth point: the author states the equivalence, the density table
     predicts it, the error is measurable. No table entry is trusted untested.
  2. Resolution rate per book: what fraction of quantity lines resolves to
     grams, by confidence class. A cup-based book must become comparable to a
     metric one, and the unresolved share is reported, not hidden.
  3. Verb-registry coverage over the corpus's method sentences, with the most
     frequent UNCOVERED verbs listed — the mining input for R1b.
  4. Procedure discrimination: the four Tiramisù methods (board card, Gisslen,
     Friberg, Larousse/Hermé) must resemble each other more than they resemble
     controls (a Suas bread, a meringue), on set+sequence similarity.

Proposal-stage: registries live inline here; they become ingredients.yaml and
techniques.yaml only in R1/R1b, if the numbers hold.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

CORPUS = Path("../canon_library")
REPORT = Path("../tests/CANON_VALIDATED_RECIPES_REPORT_v001.pdf")
OUT = Path("../evidence/T98/standards.json")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).lower()


# ---------------------------------------------------------------- gravimetry
#: g/ml — the entries under test in measurement 1.
DENSITY = {
    "flour": 0.50, "sugar": 0.85, "powdered_sugar": 0.50, "brown_sugar": 0.80,
    "butter": 0.95, "milk": 1.03, "cream": 1.00, "water": 1.00, "oil": 0.92,
    "honey": 1.42, "cocoa": 0.50, "cornstarch": 0.53, "rice": 0.85,
    "yogurt": 1.03, "salt": 1.20, "generic_liquid": 1.00,
}
#: grams per piece — measurement 1 checks the egg family where books state it.
UNIT_G = {"egg": 50.0, "egg_yolk": 18.0, "egg_white": 30.0,
          "gelatin_sheet": 2.5, "vanilla_pod": 3.0}

DENSITY_SYNONYMS = {
    "flour": ["flour", "farina", "farine"],
    "sugar": ["granulated sugar", "caster sugar", "sugar", "zucchero", "sucre"],
    "powdered_sugar": ["powdered sugar", "confectioners", "icing sugar",
                       "zucchero a velo", "sucre glace"],
    "brown_sugar": ["brown sugar"],
    "butter": ["butter", "burro", "beurre"],
    "milk": ["milk", "latte", "lait"],
    "cream": ["cream", "panna", "creme", "crème"],
    "water": ["water", "acqua", "eau"],
    "oil": ["oil", "olio", "huile"],
    "honey": ["honey", "miele", "miel"],
    "cocoa": ["cocoa", "cacao"],
    "cornstarch": ["cornstarch", "corn starch", "maizena", "amido"],
    "rice": ["rice", "riso", "riz"],
    "yogurt": ["yogurt", "yoghurt"],
    "salt": ["salt", "sale", "sel"],
}

VOL_ML = {"cup": 240.0, "cups": 240.0, "tbsp": 15.0, "tablespoon": 15.0,
          "tablespoons": 15.0, "tsp": 5.0, "teaspoon": 5.0, "teaspoons": 5.0,
          "fl oz": 30.0, "pt": 473.0, "qt": 946.0, "ml": 1.0, "cl": 10.0,
          "dl": 100.0, "l": 1000.0}
MASS_G = {"g": 1.0, "kg": 1000.0, "oz": 28.35, "lb": 453.6, "mg": 0.001}

#: Dual-unit lines come in two shapes: "name (equiv)" and — Friberg's house
#: style — "qty unit (equiv) name". Both are truth points; all unit directions
#: are classified: vol→vol validates the ml table, mass→mass the gram table,
#: vol→mass the DENSITIES, count→mass the piece weights.
_FRAC = {"¼": .25, "½": .5, "¾": .75, "⅓": 1/3, "⅔": 2/3, "⅛": .125}
_NUM = r"(\d+\s*[¼½¾⅓⅔⅛]|\d+(?:[./]\d+)?|[¼½¾⅓⅔⅛])"
_U = r"(cups?|tablespoons?|tbsp|teaspoons?|tsp|fl\s*oz|ounces?|oz|pounds?|lb|pt|qt)"
_DUAL_AFTER = re.compile(
    _NUM + r"\s*" + _U + r"\s+([a-zà-ÿ ,'-]{3,40}?)\s*"
    r"\(\s*(\d+(?:\.\d+)?)\s*(g|ml|mL|kg|l)\s*\)", re.I)
_DUAL_BEFORE = re.compile(
    _NUM + r"\s*" + _U + r"\s*\(\s*(\d+(?:\.\d+)?)\s*(g|ml|mL|kg|l)\s*\)"
    r"\s+([a-zà-ÿ ,'-]{3,40})", re.I)
_NQ = re.compile(r"\b(to taste|q\.?\s?b\.?|as needed|pinch|a piacere|au goût|"
                 r"selon le goût|quanto basta)\b", re.I)
_COUNT = re.compile(r"\b(\d{1,2})\s+(egg yolks?|yolks?|egg whites?|whites?|eggs?|"
                    r"tuorli|albumi|uova|jaunes?|blancs?|oeufs?|"
                    r"gelatin(?:e|a)? (?:sheets?|leaves|fogli)|feuilles? de gélatine)\b", re.I)


def frac_to_float(tok: str) -> float:
    tok = tok.strip()
    m = re.match(r"(\d+)\s*([¼½¾⅓⅔⅛])", tok)
    if m:
        return float(m.group(1)) + _FRAC[m.group(2)]
    if tok in _FRAC:
        return _FRAC[tok]
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b)
    return float(tok)


def density_for(name: str) -> tuple[str, float] | None:
    n = norm(name)
    for canon, syns in DENSITY_SYNONYMS.items():
        if any(s in n for s in syns):
            return canon, DENSITY[canon]
    return None


# ---------------------------------------------------------------- procedures
TECHNIQUES = {
    "WHIP": ["whip", "whisk", "beat", "montare", "monta", "sbatt", "fouett", "battre"],
    "FOLD": ["fold", "incorporar", "incorporer", "mescolare delicat", "mix carefully"],
    "MIX": ["mix", "stir", "combine", "mescolare", "mélanger", "melanger", "amalgam"],
    "KNEAD": ["knead", "impastare", "pétrir", "petrir"],
    "SIFT": ["sift", "setacciare", "tamiser", "tamis"],
    "STRAIN": ["strain", "filtrare", "passer au chinois", "chinois", "colare"],
    "BAKE": ["bake", "cuocere in forno", "infornare", "cuire au four", "oven"],
    "BOIL": ["boil", "bollire", "bouillir", "ebollizione", "ébullition"],
    "SIMMER": ["simmer", "sobbollire", "frémir", "fremir", "mijoter"],
    "COOK_SUGAR": ["cook the sugar", "cook sugar", "cuocere lo zucchero",
                   "cuire le sucre", "soft ball", "boulé", "boule", "petit boulé"],
    "FRY": ["fry", "friggere", "frire", "sauté", "saute", "rosolare", "poêler"],
    "CARAMELIZE": ["caramel", "caramell", "caramél"],
    "MELT": ["melt", "fondere", "fondre", "sciogliere", "faire fondre"],
    "REDUCE": ["reduce", "ridurre", "réduire", "reduire"],
    "INFUSE": ["infuse", "infondere", "infuser", "steep", "macerare", "macérer"],
    "CHILL": ["chill", "refriger", "raffredd", "refroidir", "réfrigér"],
    "FREEZE": ["freeze", "congelare", "congeler", "surgeler", "abbatt", "blast"],
    "BLOOM": ["bloom", "ammoll", "gonfiare la gelatina", "ramollir", "soak the gelatin",
              "soft gelatin"],
    "LAYER": ["layer", "spread", "strato", "stendere", "étaler", "etaler"],
    "SOAK": ["soak", "brush", "imbib", "inzupp", "moisten", "puncher", "imbiber"],
    "PIPE": ["pipe", "pocher con sac", "poche à douille", "sac a poche", "piping"],
    "MOLD": ["mold", "mould", "stampo", "moule", "cerchio", "ring", "frame"],
    "GLAZE": ["glaze", "glassare", "glacer", "napper", "nappare"],
    "DUST": ["dust", "spolver", "saupoudr", "sprinkle"],
    "EMULSIFY": ["emulsif", "emulsion", "émulsion"],
    "TEMPER": ["temper", "temperare", "tempérer", "temperer"],
    "PROOF": ["proof", "lievitare", "lever", "pointage", "rise"],
    "REST": ["rest", "riposare", "reposer", "far riposare"],
    "GRIND": ["grind", "tritare", "macinare", "hacher", "mixer", "blend", "frullare"],
    "SEASON": ["season", "condire", "assaisonner", "insaporire"],
    # aggiunti dal mining T98 (i primi non-coperti: put/heat/cut/add/place/remove/scale)
    "ADD": ["add ", "aggiungere", "unire", "ajouter"],
    "HEAT": ["heat", "scaldare", "riscaldare", "chauffer", "warm "],
    "CUT": ["cut ", "tagliare", "couper", "slice", "affettare", "trancher"],
    "PLACE": ["place", "put ", "transfer", "disporre", "mettere", "poser", "déposer", "deposer"],
    "REMOVE": ["remove", "togliere", "retirer", "sformare", "unmold"],
    "WEIGH": ["scale ", "weigh", "pesare", "peser", "dosare"],
    "ROLL": ["roll ", "stendere col matterello", "abaisser", "arrotolare"],
    "COVER": ["cover", "coprire", "couvrir", "wrap", "pellicola"],
    "POUR": ["pour", "versare", "verser"],
    "DRAIN": ["drain", "scolare", "égoutter", "egoutter"],
}
_STEP = re.compile(r"(?:^|\n)\s*\d{1,2}[.)]\s+([^\n]{15,300})")


def techniques_in(text: str, ordered: bool = False):
    body = norm(text)
    if not ordered:
        return {t for t, syns in TECHNIQUES.items() if any(s in body for s in syns)}
    seq = []
    for m in re.finditer(r"[^.!?\n]{10,300}[.!?\n]", body):
        sent = m.group(0)
        for t, syns in TECHNIQUES.items():
            if any(s in sent for s in syns) and (not seq or seq[-1] != t):
                seq.append(t)
    return seq


def lcs(a: list, b: list) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else \
                max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def proc_similarity(ta, sa, tb, sb, alpha=0.5) -> float:
    j = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    s = lcs(sa, sb) / max(len(sa), len(sb)) if sa and sb else 0.0
    return alpha * j + (1 - alpha) * s


# ------------------------------------------------------------------- the lab
def main() -> int:
    import fitz

    books = sorted(p for p in CORPUS.iterdir()
                   if p.suffix.lower() in (".pdf", ".epub"))
    out: dict = {"1_gravimetria_doppia_unita": {}, "2_risoluzione_per_libro": {},
                 "3_copertura_verbi": {}, "4_discriminazione_procedure": {}}

    # --- 1. densities vs the books' own dual-unit lines -------------------
    print("=" * 78)
    print("1. GRAVIMETRIA CONTRO LA VERITÀ DEI LIBRI (righe a doppia unità)")
    per_ing: dict[str, list] = {}
    checked = 0
    for f in books:
        with fitz.open(f) as doc:
            for i in range(doc.page_count):
                t = doc[i].get_text()
                hits = [(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))
                        for m in _DUAL_AFTER.finditer(t)]
                hits += [(m.group(1), m.group(2), m.group(5), m.group(3), m.group(4))
                         for m in _DUAL_BEFORE.finditer(t)]
                for qty, unit, name, stated, s_unit in hits:
                    d = density_for(name)
                    if not d:
                        continue
                    canon, rho = d
                    u = re.sub(r"\s+", " ", unit.lower())
                    stated_v = float(stated)
                    if s_unit.lower() in ("kg", "l"):
                        stated_v *= 1000.0
                        s_unit = "g" if s_unit.lower() == "kg" else "ml"
                    if u in ("ounces", "ounce", "oz", "pounds", "pound", "lb"):
                        grams = frac_to_float(qty) * MASS_G["oz" if u.startswith("o") else "lb"]
                        if s_unit.lower() == "g":
                            pred, truth, kind = grams, stated_v, "unita_massa"
                        else:
                            continue
                    else:
                        ml = frac_to_float(qty) * VOL_ML[u]
                        if s_unit.lower() == "ml":
                            pred, truth, kind = ml, stated_v, "unita_volume"
                        else:
                            pred, truth, kind = ml * rho, stated_v, "densita"
                    if truth <= 0:
                        continue
                    err = abs(pred - truth) / truth
                    per_ing.setdefault(f"{canon}:{kind}", []).append(err)
                    checked += 1
    summary = {}
    for k, errs in sorted(per_ing.items()):
        errs.sort()
        med = errs[len(errs) // 2]
        summary[k] = {"n": len(errs), "err_mediano_pct": round(med * 100, 1),
                      "err_max_pct": round(max(errs) * 100, 1)}
        flag = "  <-- FUORI SOGLIA" if med > 0.15 and k.endswith("densita") else ""
        print(f"   {k:<28} n={len(errs):>3}  err mediano {med*100:5.1f}%  "
              f"max {max(errs)*100:5.1f}%{flag}")
    dens = [v for k, v in summary.items() if k.endswith("densita")]
    med_all = sorted(x["err_mediano_pct"] for x in dens)[len(dens)//2] if dens else None
    print(f"   totale righe verificate: {checked} · mediana degli errori mediani "
          f"(densità): {med_all}%")
    out["1_gravimetria_doppia_unita"] = {"righe": checked, "per_ingrediente": summary,
                                         "mediana_densita_pct": med_all}

    # --- 2. resolution rate per book ---------------------------------------
    print("\n" + "=" * 78)
    print("2. TASSO DI RISOLUZIONE IN GRAMMI, PER LIBRO")
    qty_any = re.compile(r"\d")
    for f in books:
        cls = Counter()
        with fitz.open(f) as doc:
            step = max(1, doc.page_count // 60)
            for i in range(0, doc.page_count, step):
                for ln in doc[i].get_text().splitlines():
                    n = norm(ln)
                    if not qty_any.search(n) and not _NQ.search(n):
                        continue
                    if _NQ.search(n):
                        cls["NON_QUANTIFICATO"] += 1
                    elif re.search(r"\d\s*(g|kg|mg|oz|lb)\b", n):
                        cls["MISURATO"] += 1
                    elif re.search(r"\d\s*(ml|cl|dl|l)\b", n) or \
                            re.search(r"(cups?|tbsp|tsp|tablespoons?|teaspoons?)\b", n):
                        cls["CONVERTITO_VOLUME"] += 1
                    elif _COUNT.search(n):
                        cls["CONVERTITO_PEZZO"] += 1
        tot = sum(cls.values())
        if tot < 20:
            continue
        risolte = tot - cls["NON_QUANTIFICATO"]
        row = {k: round(100 * v / tot, 1) for k, v in cls.items()}
        row["risolte_pct"] = round(100 * risolte / tot, 1)
        out["2_risoluzione_per_libro"][f.name[:44]] = row
        print(f"   {f.name[:42]:<44} risolte {row['risolte_pct']:5.1f}%  "
              f"(masse {row.get('MISURATO', 0)}%, volumi {row.get('CONVERTITO_VOLUME', 0)}%, "
              f"pezzi {row.get('CONVERTITO_PEZZO', 0)}%, nq {row.get('NON_QUANTIFICATO', 0)}%)")

    # --- 3. verb coverage ---------------------------------------------------
    print("\n" + "=" * 78)
    print("3. COPERTURA DEL REGISTRO DEI VERBI SULLE FRASI-METODO")
    covered = total = 0
    uncovered_first_words: Counter = Counter()
    for f in books:
        with fitz.open(f) as doc:
            step = max(1, doc.page_count // 40)
            for i in range(0, doc.page_count, step):
                t = doc[i].get_text()
                for m in _STEP.finditer(t):
                    sent = norm(m.group(1))
                    total += 1
                    if any(s in sent for syns in TECHNIQUES.values() for s in syns):
                        covered += 1
                    else:
                        w = re.match(r"[a-zà-ÿ]{3,}", sent)
                        if w:
                            uncovered_first_words[w.group(0)] += 1
    pct = round(100 * covered / max(total, 1), 1)
    print(f"   frasi-metodo campionate: {total} · coperte: {covered} ({pct}%)")
    top_missing = uncovered_first_words.most_common(15)
    print(f"   verbi frequenti NON coperti (input per il mining R1b):")
    for w, n in top_missing:
        print(f"      {n:>4}  {w}")
    out["3_copertura_verbi"] = {"frasi": total, "coperte": covered, "pct": pct,
                                "non_coperti_top": top_missing}

    # --- 4. procedure discrimination ---------------------------------------
    print("\n" + "=" * 78)
    print("4. DISCRIMINAZIONE: I 4 TIRAMISÙ CONTRO 2 CONTROLLI")
    with fitz.open(REPORT) as rep:
        card = next(rep[i].get_text() for i in range(rep.page_count)
                    if "Tiramis" in rep[i].get_text())
    methods = {"scheda MSC": card.split("Method:", 1)[1]}

    def page_text(needle, pg):
        f = next(x for x in books if needle in x.name)
        with fitz.open(f) as d:
            return d[pg - 1].get_text()

    methods["Gisslen 489"] = page_text("Professional Baking", 489)
    methods["Friberg 383"] = page_text("Advanced Professional", 383)
    methods["Larousse 851"] = page_text("Larousse", 851)
    methods["CTRL pane Suas"] = page_text("Advanced bread", 283)
    methods["CTRL meringa TPC"] = page_text("Professional Chef", 795)

    feats = {k: (techniques_in(v), techniques_in(v, ordered=True))
             for k, v in methods.items()}
    names = list(feats)
    print(f"   {'':<22}" + "".join(f"{n[:12]:>14}" for n in names))
    matrix = {}
    for a in names:
        row = []
        for b in names:
            s = proc_similarity(feats[a][0], feats[a][1], feats[b][0], feats[b][1])
            row.append(s)
        matrix[a] = [round(x, 2) for x in row]
        print(f"   {a:<22}" + "".join(f"{x:>14.2f}" for x in row))
    tira = names[:4]
    intra = [matrix[a][names.index(b)] for a in tira for b in tira if a != b]
    cross = [matrix[a][names.index(b)] for a in tira for b in names[4:]]
    m_intra = sum(intra) / len(intra)
    m_cross = sum(cross) / len(cross)
    print(f"   similarità media FRA tiramisù: {m_intra:.2f} · "
          f"VERSO i controlli: {m_cross:.2f} · separazione: "
          f"{'SÌ' if m_intra > m_cross else 'NO'}")
    out["4_discriminazione_procedure"] = {
        "matrice": matrix, "media_intra_tiramisu": round(m_intra, 3),
        "media_verso_controlli": round(m_cross, 3),
        "separa": m_intra > m_cross}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
