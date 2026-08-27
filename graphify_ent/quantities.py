"""The programme's ONE quantity grammar (ADR-0004, Q2).

Promoted out of the recipe package so that verify, the recipe layer and the
coming canonical fact layer all read figures the same way. Before this module
there were two disagreeing unit tables in-tree (`verify._TO_BASE` and
`ingredients.yaml`) and a third was about to ship; a figure one table knew and
another did not was the verify_claim defect: "1 gal" fell through to a
substring search and was confirmed off a yield line. One grammar, one table —
the YAML — and a figure the grammar cannot read is REFUSED downstream, never
guessed (ADR-0004 Q4).

What lives here: numbers as books write them (mixed numbers, ASCII and vulgar
fractions, comma decimals), ranges ("2 to 3 qt", "da 165 a 180 °C"), dual
notations ("5 qt/4.80 L" is two statements of one figure), unit tables loaded
from the pluggable YAML with conversion to three base units (g, ml, °C), and
the four declared resolution classes with their uncertainty. What does NOT
live here: ingredient identity, synonyms, densities — those are vocabulary,
they stay in the registry's YAML per the DOMAIN-AGNOSTIC rule.

`scan` works on `norm`-alised text and its spans refer to that normalised
string; `norm` is public precisely so callers can align. NFKD is load-bearing:
it takes "½" apart into "1⁄2" with a FRACTION SLASH, and mapping that slash to
an ordinary one is what lets the number parser read a half instead of seeing
a bare "2" against the unit — measured, "½ cup" was read as "2 cup".
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = [
    "CONVERTED_PIECE",
    "CONVERTED_VOLUME",
    "MEASURED",
    "UNQUANTIFIED",
    "UNCERTAINTY",
    "FRACTIONS",
    "NUM",
    "Quantity",
    "UnitTable",
    "default_table",
    "norm",
    "norm_spans",
    "number",
    "to_original",
    "scan",
]

MEASURED = "MEASURED"
CONVERTED_VOLUME = "CONVERTED_VOLUME"
CONVERTED_PIECE = "CONVERTED_PIECE"
UNQUANTIFIED = "UNQUANTIFIED"

#: Declared uncertainty per class. A converted figure is not a measured one and
#: the score has to be able to say so; hiding the difference is how a table of
#: approximations turns into false precision.
UNCERTAINTY = {MEASURED: 0.0, CONVERTED_VOLUME: 0.15,
               CONVERTED_PIECE: 0.10, UNQUANTIFIED: 1.0}

#: Units the YAML declares plus the temperature scales; everything converts to
#: exactly one of these.
BASE_UNITS = ("g", "ml", "c")

FRACTIONS = {"¼": 0.25, "½": 0.5, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3,
             "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875}

#: A quantity as books write it: "2", "1.5", "1,5", "1/2", "1 1/2". The mixed
#: number comes FIRST in the alternation or "1 1/2" is read as a bare 1.
NUM = (r"\d+\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+|"
       r"\d+\s*[¼½¾⅓⅔⅛⅜⅝⅞]|\d+(?:[.,]\d+)?|[¼½¾⅓⅔⅛⅜⅝⅞]")

#: Separators a range is written with. "da 165 a 180 °C" and "2 to 3 qt" are
#: one statement with two endpoints, not two figures.
_RANGE_SEP = r"(?:to|a|–|—|-)"

#: The YAML whose `units:` section is the single surviving table (ADR-0004).
DEFAULT_UNITS = Path(__file__).with_name("recipes") / "ingredients.yaml"


def norm(s: str) -> str:
    """Accent- and case-insensitive form. The corpus holds three languages and
    the same word in two spellings; comparing raw text loses half of it.

    NFKD also takes vulgar fractions apart — "½" becomes "1⁄2" with a FRACTION
    SLASH — and that quietly changed a quantity: "½ cup" was read as "2 cup",
    because the denominator is a digit sitting next to the unit. Mapping the
    fraction slash to an ordinary one turns it back into something the number
    parser reads as a half.
    """
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("⁄", "/")
    return re.sub(r"\s+", " ", s).strip().lower()


def norm_spans(text: str) -> tuple[str, list[int]]:
    """`norm(text)` plus, per normalised character, the index of the original
    character it came from.

    `scan` works on the normalised string, but evidence must be quoted from
    the ORIGINAL — `verify_evidence_binding` collapses whitespace and case,
    not accents, so a folded "bechamel" would fail the binding check against a
    page that writes "béchamel". This map is what lets a span found in the
    normalised text be cut, verbatim, from the source passage.
    """
    out: list[str] = []
    idx: list[int] = []
    ws_at = -1
    for i, ch in enumerate(text or ""):
        if ch.isspace():
            if ws_at < 0:
                ws_at = i
            continue
        kept = [c for c in unicodedata.normalize("NFKD", ch)
                if not unicodedata.combining(c)]
        if not kept:
            ws_at = -1 if ws_at < 0 else ws_at
            continue
        if ws_at >= 0 and out:
            out.append(" ")
            idx.append(ws_at)
        ws_at = -1
        for c in kept:
            out.append("/" if c == "⁄" else c.lower())
            idx.append(i)
    return "".join(out), idx


def to_original(span: tuple[int, int], idx: list[int]) -> tuple[int, int]:
    """A span in the normalised string to the slice of the original text."""
    s, e = span
    if not idx or s >= len(idx) or e <= s:
        return (0, 0)
    e = min(e, len(idx))
    return idx[s], idx[e - 1] + 1


def number(tok: str) -> float | None:
    """One numeric token to a float, or None — never a guess."""
    tok = (tok or "").strip().replace(",", ".")
    m = re.match(r"^(\d+)\s*([¼½¾⅓⅔⅛⅜⅝⅞])$", tok)
    if m:
        return float(m.group(1)) + FRACTIONS[m.group(2)]
    if tok in FRACTIONS:
        return FRACTIONS[tok]
    m = re.match(r"^(\d+)\s+(\d+)\s*/\s*(\d+)$", tok)      # "1 1/2"
    if m:
        try:
            return float(m.group(1)) + float(m.group(2)) / float(m.group(3))
        except ZeroDivisionError:
            return None
    if "/" in tok:
        a, _, b = tok.partition("/")
        try:
            return float(a.strip()) / float(b.strip())
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(tok)
    except ValueError:
        return None


@dataclass(frozen=True)
class Quantity:
    """One figure, read and converted. `lo == hi` unless the source wrote a
    range. `raw` is the exact matched text — the quotable original — and
    `span` locates it in the NORMALISED string the scan ran on."""

    lo: float
    hi: float
    unit: str            # one of BASE_UNITS
    raw: str
    span: tuple[int, int]

    @property
    def single(self) -> bool:
        return self.lo == self.hi

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2


class UnitTable:
    """The unit vocabulary, loaded from YAML and never hardcoded — the same
    pluggability rule the ingredient registry lives by."""

    def __init__(self, volume_ml: dict | None = None, mass_g: dict | None = None,
                 temperature: dict | None = None, version: int | str = 0):
        self.version = version
        self.volume_ml = {norm(k): float(v) for k, v in (volume_ml or {}).items()}
        self.mass_g = {norm(k): float(v) for k, v in (mass_g or {}).items()}
        #: token -> scale letter ("c"/"f"); the °F formula needs an offset, so
        #: it is applied in `to_base`, not stored as a factor.
        self.temperature = {norm(k): str(v).strip().lower()
                            for k, v in (temperature or {}).items()}
        tokens = sorted(set(self.volume_ml) | set(self.mass_g) | set(self.temperature),
                        key=len, reverse=True)
        self._alt = "|".join(re.escape(t) for t in tokens)
        joined = self._alt or r"(?!x)x"          # match nothing when empty
        self._single = re.compile(rf"({NUM})\s*({joined})(?![a-z])", re.I)
        self._range = re.compile(
            rf"({NUM})\s*{_RANGE_SEP}\s*({NUM})\s*({joined})(?![a-z])", re.I)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "UnitTable":
        import yaml

        p = Path(path or os.environ.get("ENTERPRIPHY_INGREDIENTS") or DEFAULT_UNITS)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        units = data.get("units", {})
        return cls(units.get("volume_ml"), units.get("mass_g"),
                   units.get("temperature"), version=data.get("version", 0))

    def to_base(self, value: float, unit: str) -> tuple[float, str]:
        """One (value, unit) to (value, base unit). Raises KeyError on a unit
        the table does not know — callers refuse, they do not fall through."""
        u = norm(unit)
        if u in self.temperature:
            if self.temperature[u] == "f":
                return (value - 32) * 5 / 9, "c"
            return value, "c"
        if u in self.mass_g:
            return value * self.mass_g[u], "g"
        return value * self.volume_ml[u], "ml"


@lru_cache(maxsize=4)
def _cached_table(path: str | None) -> UnitTable:
    return UnitTable.from_yaml(path)


def default_table() -> UnitTable:
    return _cached_table(os.environ.get("ENTERPRIPHY_INGREDIENTS"))


def scan(text: str, table: UnitTable | None = None) -> list[Quantity]:
    """Every figure in `text`, converted to base units, in reading order.

    Ranges are matched first and their spans are consumed, so "2 to 3 qt" is
    one Quantity with two endpoints and never also a bare "3 qt". Dual
    notations ("5 qt/4.80 L", "165°F/74°C") come out as two Quantities — two
    statements of one figure, and the caller can see they agree. Text the
    grammar cannot read yields NOTHING; absence is the refusal signal
    (ADR-0004 Q4), there is no partial guess.
    """
    t = table or default_table()
    body = norm(text)
    out: list[Quantity] = []
    taken: list[tuple[int, int]] = []
    for m in t._range.finditer(body):
        a, b = number(m.group(1)), number(m.group(2))
        if a is None or b is None:
            continue
        va, ua = t.to_base(a, m.group(3))
        vb, _ = t.to_base(b, m.group(3))
        out.append(Quantity(min(va, vb), max(va, vb), ua, m.group(0), m.span()))
        taken.append(m.span())
    for m in t._single.finditer(body):
        if any(s < m.end() and m.start() < e for s, e in taken):
            continue
        v = number(m.group(1))
        if v is None:
            continue
        vb, ub = t.to_base(v, m.group(2))
        out.append(Quantity(vb, vb, ub, m.group(0), m.span()))
    out.sort(key=lambda q: q.span)
    return out
