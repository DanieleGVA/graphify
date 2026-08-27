"""The T98 technique registry, promoted as it was validated — mining deferred.

The revised plan (2026-08-27) ships the 40-verb registry exactly as
`tools/standards_lab.py` measured it: 70.6% coverage of the corpus's method
sentences after one mining pass. Procedure carries 0.25 of the match score and
T96 showed ingredients alone put the target 3rd of 1,578 — so the mining loop
to >=90% reopens only if R5's failures say the procedure criterion is what is
missing, not before.

Sequence matters as well as membership: pâte à bombe and Italian meringue use
almost the same verbs in a different order. `similarity` therefore blends
Jaccard on the set with normalised LCS on the sequence.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["TECHNIQUES", "lcs", "similarity", "techniques_in"]

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
    "BLOOM": ["bloom", "ammoll", "gonfiare la gelatina", "ramollir",
              "soak the gelatin", "soft gelatin"],
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
    "ADD": ["add ", "aggiungere", "unire", "ajouter"],
    "HEAT": ["heat", "scaldare", "riscaldare", "chauffer", "warm "],
    "CUT": ["cut ", "tagliare", "couper", "slice", "affettare", "trancher"],
    "PLACE": ["place", "put ", "transfer", "disporre", "mettere", "poser",
              "déposer", "deposer"],
    "REMOVE": ["remove", "togliere", "retirer", "sformare", "unmold"],
    "WEIGH": ["scale ", "weigh", "pesare", "peser", "dosare"],
    "ROLL": ["roll ", "stendere col matterello", "abaisser", "arrotolare"],
    "COVER": ["cover", "coprire", "couvrir", "wrap", "pellicola"],
    "POUR": ["pour", "versare", "verser"],
    "DRAIN": ["drain", "scolare", "égoutter", "egoutter"],
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).lower()


def techniques_in(text: str, ordered: bool = False):
    """Canonical techniques a text uses — as a set, or as a sequence.

    The sequence walks sentence by sentence and drops immediate repeats, so
    "whip... keep whipping" is one WHIP, not two.
    """
    body = _norm(text)
    if not ordered:
        return {t for t, syns in TECHNIQUES.items() if any(s in body for s in syns)}
    seq: list[str] = []
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


def similarity(ta: set, sa: list, tb: set, sb: list, alpha: float = 0.5) -> float:
    """Jaccard on the technique set, LCS on the sequence, blended.

    "Whip then fold" is not "fold then whip": pâte à bombe and Italian meringue
    share their verbs and differ in the order, which is why membership alone
    cannot carry the criterion.
    """
    j = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    s = lcs(sa, sb) / max(len(sa), len(sb)) if sa and sb else 0.0
    return alpha * j + (1 - alpha) * s
