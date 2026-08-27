"""R1 — canonical ingredients and the gravimetric standard.

An ingredient line becomes `(canonical name, grams, how we know)`. Everything a
recipe is compared on rests on this, so the two things it must never do are
guess silently and lose the reason.

**One unit.** Proportions need a single unit, and the unit is the gram. Every
line resolves into one of four declared classes, never a silent conversion:

    MEASURED           mass stated by the book            "450 g"        exact
    CONVERTED_VOLUME   volume x the ingredient's density  "1 cup flour"  tabular
    CONVERTED_PIECE    count x a standard piece weight    "3 egg yolks"  tabular
    UNQUANTIFIED       "to taste", "q.b.", "a pinch"                     none

UNQUANTIFIED counts for PRESENCE — which under rarity weighting is already half
of a recipe's identity — and never for the proportions.

**Density belongs to the ingredient in its state, not to the unit.** A cup of
flour is 120 g, of sugar 200 g, of honey 340 g; and whipping cream halves its
density, which is why `cream` and `cream_whipped` are two entries. Taking one
for the other was a 150% error when it was measured. The table is versioned in
`ingredients.yaml`, checked against the books' own dual-unit lines ("¾ cup
(180 ml)"), and its median error there is 6.2% (evidence/T98).

**Three parsing rules, each bought with a measured failure** (evidence/T96):

  * a quantity belongs to the NEAREST PRECEDING ingredient line, one to one. The
    first version took any quantity within two lines, so the section header
    "COFFEE SYRUP" absorbed the 2,000 g of the water beneath it — and the water
    kept it too. The query became a sweet coffee liquid and its best match in
    1,743 pages was a BBQ sauce.
  * an ingredient line that receives no quantity before the next ingredient line
    receives nothing. That is exactly what a section header is.
  * the LEFTMOST synonym in a line decides the canonical, because an ingredient
    line leads with its head noun. First-match-wins classified "CREAM WHIPPING
    36% UHT SUGAR FREE" as sugar — and the cream vanished.

No model is called. A wrong reading must be explainable by looking at the line.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# The number grammar, the four resolution classes and `norm` were promoted to
# graphify_ent.quantities (ADR-0004 Q2): ONE grammar for the recipe layer,
# verify and the canonical fact layer. Re-imported here so R1's public API and
# every existing caller stay exactly as they were.
from graphify_ent.quantities import (
    CONVERTED_PIECE,
    CONVERTED_VOLUME,
    MEASURED,
    UNCERTAINTY,
    UNQUANTIFIED,
    FRACTIONS as _FRACTIONS,
    NUM as _NUM,
    norm,
    number as _to_float,
)

__all__ = [
    "CONVERTED_PIECE",
    "CONVERTED_VOLUME",
    "MEASURED",
    "UNQUANTIFIED",
    "DEFAULT_REGISTRY",
    "Registry",
    "Resolved",
    "norm",
    "parse_block",
    "proportions",
    "resolve_line",
]

DEFAULT_REGISTRY = Path(__file__).with_name("ingredients.yaml")

#: How many lines one ingredient row may span. A flattened table row is name +
#: US column + metric column; beyond that the "row" has swallowed prose.
MAX_ROW_LINES = 4

#: Units a book prints as its working figure when it prints two systems.
_METRIC_MASS = {"mg", "g", "gr", "kg"}


@dataclass(frozen=True)
class Resolved:
    """One ingredient line, read.

    `grams` is None only for UNQUANTIFIED — a line that names an ingredient
    without stating how much. It is still a Resolved, because presence is
    evidence and dropping it would silently shrink the recipe.
    """

    canonical: str
    grams: float | None
    cls: str
    raw: str
    uncertainty: float = 0.0

    @property
    def quantified(self) -> bool:
        return self.grams is not None and self.cls != UNQUANTIFIED

    def as_dict(self) -> dict:
        return {"canonical": self.canonical, "grams": self.grams,
                "class": self.cls, "raw": self.raw,
                "uncertainty": self.uncertainty}


class Registry:
    """The vocabulary, loaded from YAML and never hardcoded.

    `ENTERPRIPHY_INGREDIENTS` points at another file; the DOMAIN-AGNOSTIC rule
    means nothing in this module may know an ingredient by name.
    """

    def __init__(self, data: dict):
        self.version = data.get("version", 0)
        self.volume_ml = {norm(k): float(v)
                          for k, v in (data.get("units", {}).get("volume_ml") or {}).items()}
        self.mass_g = {norm(k): float(v)
                       for k, v in (data.get("units", {}).get("mass_g") or {}).items()}
        self.unquantified = [norm(p) for p in (data.get("unquantified") or [])]
        self.ingredients: dict[str, dict] = {}
        #: (normalised synonym, canonical), longest first — a longer synonym is
        #: a more specific one, and "egg white" must win over "egg" wherever
        #: both could match at the same position.
        pairs: list[tuple[str, str]] = []
        for canon, entry in (data.get("ingredients") or {}).items():
            entry = entry or {}
            self.ingredients[canon] = entry
            for syn in (entry.get("synonyms") or [canon]):
                pairs.append((norm(syn), canon))
        self._synonyms = sorted(pairs, key=lambda p: -len(p[0]))

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Registry":
        import yaml

        p = Path(path or os.environ.get("ENTERPRIPHY_INGREDIENTS") or DEFAULT_REGISTRY)
        return cls(yaml.safe_load(p.read_text(encoding="utf-8")) or {})

    # -- vocabulary --------------------------------------------------------
    def canonical(self, text: str) -> str | None:
        """The ingredient a line is about, decided by its LEFTMOST synonym.

        Not the first synonym that matches anywhere: an ingredient line leads
        with its head noun, and trade names trail behind it. "CREAM WHIPPING
        36% UHT SUGAR FREE" is cream, and first-match-wins called it sugar.
        Ties at the same position go to the longer synonym, so "egg white" is
        not read as "egg".
        """
        body = norm(text)
        if not body:
            return None
        best_at, best_canon, best_len = None, None, 0
        for syn, canon in self._synonyms:
            at = body.find(syn)
            if at < 0:
                continue
            if best_at is None or at < best_at or (at == best_at and len(syn) > best_len):
                best_at, best_canon, best_len = at, canon, len(syn)
        return best_canon

    def density(self, canonical: str) -> float | None:
        d = (self.ingredients.get(canonical) or {}).get("density")
        return float(d) if d is not None else None

    def unit_g(self, canonical: str) -> float | None:
        u = (self.ingredients.get(canonical) or {}).get("unit_g")
        return float(u) if u is not None else None

    def is_unquantified(self, text: str) -> bool:
        body = norm(text)
        return any(p in body for p in self.unquantified)

    # -- quantities --------------------------------------------------------
    def quantities(self, text: str) -> list[tuple[float, str, str]]:
        """Every `(value, unit, kind)` a line states, in order.

        `kind` is "mass" or "volume". Counts are handled separately, because a
        bare number is only a quantity when the ingredient has a piece weight —
        "2 tomatoes" is a count, "2 g" is not.
        """
        out: list[tuple[float, str, str]] = []
        units = "|".join(sorted((set(self.mass_g) | set(self.volume_ml)),
                                key=len, reverse=True))
        if not units:
            return out
        pattern = re.compile(rf"({_NUM})\s*({units})(?![a-z])", re.I)
        for m in pattern.finditer(norm(text)):
            value = _to_float(m.group(1))
            if value is None:
                continue
            unit = norm(m.group(2))
            if unit in self.mass_g:
                out.append((value, unit, "mass"))
            else:
                out.append((value, unit, "volume"))
        return out

    def count(self, text: str, canonical: str | None = None) -> float | None:
        """How many pieces a line states, for lines that number rather than weigh.

        A count sits against the thing counted, and not always at the start of
        the line: a flattened table row reads "Egg yolks 2 yolks 2 yolks",
        where the name comes first and the figure twice after it — once per
        column. So look for a number followed by a synonym of THIS ingredient,
        and fall back to a leading number for the plain "3 eggs" shape.
        """
        body = norm(text)
        if canonical:
            syns = sorted((s for s, c in self._synonyms if c == canonical),
                          key=len, reverse=True)
            if syns:
                alt = "|".join(re.escape(s) for s in syns)
                m = re.search(rf"({_NUM})\s+(?:{alt})\b", body)
                if m:
                    return _to_float(m.group(1))
        m = re.match(rf"^\s*({_NUM})\s+(?![a-z]*\d)", body)
        return _to_float(m.group(1)) if m else None


@lru_cache(maxsize=4)
def _cached(path: str | None) -> Registry:
    return Registry.load(path)


def _registry(registry: Registry | None) -> Registry:
    return registry or _cached(os.environ.get("ENTERPRIPHY_INGREDIENTS"))


def resolve_line(line: str, registry: Registry | None = None) -> Resolved | None:
    """Read one ingredient line into grams, or say why it could not be.

    Returns None when the line names no ingredient this registry knows — the
    caller reports those rather than dropping them, because an unknown line is
    a gap in the vocabulary and the exit criterion of R1 is measured coverage,
    not the length of a list.
    """
    reg = _registry(registry)
    canon = reg.canonical(line)
    if not canon:
        return None

    qty = reg.quantities(line)
    # A book that prints both systems gives the US column and the metric one
    # on the same line ("Sugar 1 lb 500 g"). The metric figure is the one to
    # take — it is what the book works in and what the T98 measurement was
    # validated against — and taking "the last" instead would read
    # "500 g flour plus 50 g for dusting" as a 50 g recipe. Mass beats volume
    # for a plainer reason: a mass needs no table to become grams.
    masses = [q for q in qty if q[2] == "mass"]
    volumes = [q for q in qty if q[2] == "volume"]
    if masses:
        metric = [q for q in masses if q[1] in _METRIC_MASS]
        if metric:
            # The LAST metric figure, because that is where a recipe puts its
            # working quantity — at the end of the row in a card
            # ("COFFEE INSTANT 200G NESCAFE 60 g") and in the metric column of
            # a two-column table ("Sugar 6 oz 180 g"). Taking the first read
            # the pack size out of the trade name and called it the recipe.
            value, unit, _ = metric[-1]
            grams = value * reg.mass_g[unit]
        else:
            # No metric column: US masses compound rather than compete —
            # "1 lb 8 oz" is one weight written in two units, not two weights.
            grams = sum(v * reg.mass_g[u] for v, u, _ in masses)
        return Resolved(canon, grams, MEASURED, line.strip(),
                        UNCERTAINTY[MEASURED])
    if volumes:
        value, unit, _ = volumes[0]
        ml = value * reg.volume_ml[unit]
        density = reg.density(canon)
        if density is not None:
            return Resolved(canon, ml * density, CONVERTED_VOLUME, line.strip(),
                            UNCERTAINTY[CONVERTED_VOLUME])
        # A volume with no density for this ingredient is not a silent 1.0:
        # say the quantity is unusable rather than invent a substance.
        return Resolved(canon, None, UNQUANTIFIED, line.strip(),
                        UNCERTAINTY[UNQUANTIFIED])
    piece = reg.unit_g(canon)
    if piece is not None:
        n = reg.count(line, canon)
        if n is not None:
            return Resolved(canon, n * piece, CONVERTED_PIECE, line.strip(),
                            UNCERTAINTY[CONVERTED_PIECE])
    # Named without a usable quantity — "salt, to taste", a section header, a
    # count of something that has no piece weight. Kept, never dropped:
    # presence is evidence, and a dropped line silently shrinks the recipe.
    return Resolved(canon, None, UNQUANTIFIED, line.strip(),
                    UNCERTAINTY[UNQUANTIFIED])


def parse_block(text: str, registry: Registry | None = None) -> list[Resolved]:
    """Read an ingredient block, obeying the one-to-one rule.

    A quantity that stands on its own line belongs to the nearest ingredient
    line ABOVE it, and only to one. An ingredient line that reaches the next
    ingredient line without collecting a quantity keeps none — which is how a
    section header ("COFFEE SYRUP") stops swallowing the figure beneath it.
    """
    reg = _registry(registry)
    rows: list[list[str]] = []
    current: list[str] = []
    current_canon: str | None = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        canon = reg.canonical(line)
        # A row ends when a DIFFERENT ingredient is named. Lines that name none
        # — a bare "180 g" — and lines that name the SAME one belong to the row
        # already open, which is what a two-column table looks like once the
        # PDF has been flattened into lines:
        #
        #     Sugar        <- the row opens
        #     6 oz         <- the US column
        #     180 g        <- the metric column, same row
        #     Water        <- a different ingredient: the row closes
        #
        # Without this, "Egg yolks / 2 yolks / 2 yolks" is read as two separate
        # ingredients and the recipe silently doubles.
        # A line naming the SAME canonical can be two different things, and
        # the difference is whether it carries its own mass or volume:
        #   "2 yolks" under "Egg yolks"      -> the other COLUMN of one row
        #   "CREAM ... 580 g" twice          -> two REAL lines, 580 g each
        # Merging both halved the cream of a real card (580 for 1,160);
        # splitting both doubled Gisslen's egg yolks. A count alone never
        # opens a row — counts are how table columns repeat.
        own_mass = any(k == "mass" or k == "volume" for _, _, k in reg.quantities(line))
        if canon and (canon != current_canon or own_mass):
            if current:
                rows.append(current)
            current, current_canon = [line], canon
        elif current and len(current) < MAX_ROW_LINES:
            current.append(line)
        elif canon:
            if current:
                rows.append(current)
            current, current_canon = [line], canon
    if current:
        rows.append(current)

    out: list[Resolved] = []
    for row in rows:
        joined = " ".join(row)
        resolved = resolve_line(joined, reg)
        if resolved is None:
            continue
        out.append(Resolved(resolved.canonical, resolved.grams, resolved.cls,
                            joined, resolved.uncertainty))
    return out


def proportions(resolved: list[Resolved]) -> dict[str, float]:
    """Grams per canonical ingredient, normalised to sum to 1.

    Scale-invariant by construction, which is the property the whole matching
    idea rests on: a recipe for ten and the same recipe for a hundred are the
    same recipe. Ingredients that carry no usable quantity are absent here and
    present in the SET — presence and proportion are two different pieces of
    evidence and the score weighs them separately.
    """
    totals: dict[str, float] = {}
    for r in resolved:
        if r.quantified:
            totals[r.canonical] = totals.get(r.canonical, 0.0) + float(r.grams)
    mass = sum(totals.values())
    if mass <= 0:
        return {}
    return {k: v / mass for k, v in totals.items()}
