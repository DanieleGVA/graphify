"""Reading a FOODMDM Pareto recipe card.

The new card export (2026-08-27) replaces the old validated-recipes report as
the test input, and it is a different kind of document. The old one was prose
that had to be guessed at; this one is a **table with declared columns**:

    #  Item code  Ingredient — verbatim  Qty  Unit  Preparation  Waste  Allergens

so the quantity and its unit are stated rather than inferred. Everything the
generic block parser had to work out — which figure belongs to which line,
which of two columns is the working one, whether "200G" is a pack size or a
quantity — is simply given here. This module reads the columns; the registry
still supplies the canonical identity and the gravimetric conversion.

What the card also carries, and the old report did not:

  * `Canon <book>` — WHEN PRESENT, the reference work the dish is supposed to
    follow. The Pareto export carried it; the merged export (2026-08-27) does
    not, and that is the point of the exercise: with no declared reference, the
    matcher has to FIND one by the three criteria and say so with evidence, or
    refuse. `canon` is then empty and `canon_available` is meaningless — the
    caller must not read absence as availability.
  * item codes, waste percentages, derived allergens, menu slots. None of them
    feed the fingerprint today; they are kept on the parsed card so a later
    stage can use them without re-reading the PDF.

Verbatim in, verbatim out: nothing here corrects, renames or merges what the
export says, per the document's own read-only rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from graphify_ent.recipes.ingredients import (
    CONVERTED_PIECE,
    CONVERTED_VOLUME,
    MEASURED,
    UNCERTAINTY,
    UNQUANTIFIED,
    Registry,
    Resolved,
    proportions,
)

__all__ = ["Card", "CardLine", "load_cards", "parse_cards"]

#: A card opens with "<n>. <title>" at the start of a line and is followed,
#: within a few lines, by its recipe code. Procedure steps are numbered too,
#: which is why the code is what confirms a real card header.
_HEADER = re.compile(r"^(\d{1,4})\.\s+(.+)$", re.M)
_CODE = re.compile(r"^(RF\d+)\s*$", re.M)
_ITEM = re.compile(r"^(CM\d+|SF\d+|RF\d+|—)$")
_NUM = re.compile(r"^\d{1,3}$")
_QTY = re.compile(r"^\d+(?:[.,]\d+)?$")
_PAGE_NOISE = re.compile(r"^(page \d+|FOODMDM ·.*|#|Item code|Ingredient(?: — verbatim)?|"
                         r"Qty|Unit|Preparation|Wastage|Waste|"
                         r"Allergens \(derived\))$")

#: Units the export uses that are not in the registry's tables: a count of
#: pieces, and "TT" (to taste) written as a unit.
_PIECE_UNITS = {"pz", "pc", "pcs", "ea"}
_TASTE_UNITS = {"tt"}


@dataclass
class CardLine:
    """One row of the ingredient table, exactly as printed."""

    index: int
    item_code: str
    ingredient: str
    qty: float | None
    unit: str
    preparation: str = ""
    waste: str = ""
    allergens: str = ""

    def resolve(self, registry: Registry) -> Resolved | None:
        """The row as grams, using the DECLARED unit — never a guessed one."""
        canonical = registry.canonical(self.ingredient)
        if not canonical:
            return None
        raw = f"{self.ingredient} {self.qty or ''} {self.unit}".strip()
        unit = (self.unit or "").strip().lower()
        if self.qty is None or unit in _TASTE_UNITS or not unit:
            return Resolved(canonical, None, UNQUANTIFIED, raw,
                            UNCERTAINTY[UNQUANTIFIED])
        if unit in registry.mass_g:
            return Resolved(canonical, self.qty * registry.mass_g[unit],
                            MEASURED, raw, UNCERTAINTY[MEASURED])
        if unit in registry.volume_ml:
            density = registry.density(canonical)
            if density is None:
                return Resolved(canonical, None, UNQUANTIFIED, raw,
                                UNCERTAINTY[UNQUANTIFIED])
            return Resolved(canonical, self.qty * registry.volume_ml[unit] * density,
                            CONVERTED_VOLUME, raw, UNCERTAINTY[CONVERTED_VOLUME])
        if unit in _PIECE_UNITS:
            piece = registry.unit_g(canonical)
            if piece is not None:
                return Resolved(canonical, self.qty * piece, CONVERTED_PIECE, raw,
                                UNCERTAINTY[CONVERTED_PIECE])
        return Resolved(canonical, None, UNQUANTIFIED, raw,
                        UNCERTAINTY[UNQUANTIFIED])


@dataclass
class Card:
    number: int
    title: str
    code: str = ""
    canon: str = ""
    domain: str = ""
    lines: list[CardLine] = field(default_factory=list)
    procedure: str = ""

    @property
    def canon_available(self) -> bool:
        """False when the card's reference work is not in the corpus.

        The export says so itself — "Italian canon — TO ACQUIRE" — which makes
        these cards the benchmark's honest negatives rather than a judgement
        call of ours.
        """
        return "TO ACQUIRE" not in (self.canon or "").upper()

    def resolved(self, registry: Registry) -> list[Resolved]:
        out = []
        for line in self.lines:
            r = line.resolve(registry)
            if r is not None:
                out.append(r)
        return out

    def proportions(self, registry: Registry) -> dict[str, float]:
        return proportions(self.resolved(registry))

    def as_dict(self) -> dict:
        return {"number": self.number, "title": self.title, "code": self.code,
                "canon": self.canon, "canon_available": self.canon_available,
                "domain": self.domain, "lines": len(self.lines),
                "has_procedure": bool(self.procedure)}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("—", "").strip())


def parse_cards(text: str) -> list[Card]:
    """Every card in the export's text, columns and all."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Card boundaries: a "<n>. <title>" line whose next few lines hold an RF
    # code. Procedure steps look the same and have no code, which is the whole
    # reason the code is required rather than assumed.
    starts: list[tuple[int, int, str]] = []
    expected = 1
    for i, ln in enumerate(lines):
        m = _HEADER.match(ln)
        if not m:
            continue
        # A procedure step is numbered exactly like a card header, and the
        # next card's RF code sits a few lines below it — so the code alone
        # promoted "2. FOR SERVING AND GARNISH:" to a card. What only a real
        # header has is its metadata line.
        window = "\n".join(lines[i + 1: i + 6])
        # A procedure step is numbered exactly like a card header and the next
        # card's RF code sits a few lines below it, so the code alone promoted
        # "2. FOR SERVING AND GARNISH:" to a card. Cards are numbered in one
        # ascending run (1..N) while procedure steps restart at 1 inside every
        # card, and that is the exact discriminator.
        n = int(m.group(1))
        # "Record type" is on the metadata line of BOTH exports; "Menu slots"
        # was only on the Pareto one, and requiring it made the reader blind to
        # the merged export entirely.
        if _CODE.search(window) and "Record type" in window and n == expected:
            starts.append((i, n, m.group(2).strip()))
            expected += 1

    cards: list[Card] = []
    for pos, (at, number, title) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        block = lines[at:end]
        body = "\n".join(block)
        # A title can wrap onto the next line before the code appears.
        code_at = next((j for j, ln in enumerate(block) if _CODE.match(ln)), None)
        if code_at and code_at > 1:
            title = " ".join([title] + [b.strip() for b in block[1:code_at]
                                        if b.strip() and not _PAGE_NOISE.match(b.strip())])
        card = Card(number=number, title=_clean(title),
                    code=_CODE.search(body).group(1) if _CODE.search(body) else "")
        canon = re.search(r"Canon\s+([^·\n]+)", body)
        card.canon = canon.group(1).strip() if canon else ""
        dom = re.search(r"Domain\s+(\S+)", body)
        card.domain = dom.group(1) if dom else ""
        proc = body.split("PROCEDURE", 1)
        card.procedure = proc[1].strip() if len(proc) > 1 else ""
        card.lines = _parse_table(proc[0].splitlines())
        cards.append(card)
    return cards


def _parse_table(block: list[str]) -> list[CardLine]:
    """The ingredient table, read as the column sequence it is.

    Each row is: index, item code, one or two name lines, quantity, unit, then
    preparation / waste / allergens. Reading it as a state machine rather than
    by position is what makes a wrapped ingredient name harmless.
    """
    rows: list[CardLine] = []
    cells = [c.strip() for c in block if c.strip() and not _PAGE_NOISE.match(c.strip())]
    i = 0
    while i < len(cells):
        if not _NUM.match(cells[i]) or i + 1 >= len(cells) or not _ITEM.match(cells[i + 1]):
            i += 1
            continue
        index, code = int(cells[i]), cells[i + 1]
        j = i + 2
        name_parts: list[str] = []
        while j < len(cells) and not _QTY.match(cells[j].replace(",", ".")):
            # The export prints an em dash wherever a column is empty — Qty
            # included — so the scan must also stop when the NEXT row starts,
            # or one unquantified row swallows the rest of the table.
            if _NUM.match(cells[j]) and j + 1 < len(cells) and _ITEM.match(cells[j + 1]):
                break
            name_parts.append(cells[j])
            j += 1
            if len(name_parts) > 5:          # runaway: not a table row after all
                break
        if not name_parts:
            i += 1
            continue
        if j >= len(cells) or not _QTY.match(cells[j].replace(",", ".")):
            rows.append(CardLine(index=index, item_code=code,
                                 ingredient=" ".join(name_parts), qty=None, unit=""))
            i = j
            continue
        qty_raw = cells[j].replace(",", ".")
        # "1,500" in this export is one thousand five hundred grams, not 1.5:
        # the column is an integer with a thousands separator. A decimal comma
        # would leave two digits, not three.
        qty = float(qty_raw) if "." not in qty_raw else (
            float(qty_raw.replace(".", "")) if len(qty_raw.split(".")[1]) == 3
            else float(qty_raw))
        unit = cells[j + 1] if j + 1 < len(cells) else ""
        rest = cells[j + 2: j + 5]
        rows.append(CardLine(index=index, item_code=code,
                             ingredient=" ".join(name_parts), qty=qty,
                             unit=_clean(unit),
                             preparation=_clean(rest[0]) if len(rest) > 0 else "",
                             waste=_clean(rest[1]) if len(rest) > 1 else "",
                             allergens=_clean(rest[2]) if len(rest) > 2 else ""))
        i = j + 2
    return rows


def load_cards(path: Path) -> list[Card]:
    import fitz

    with fitz.open(path) as doc:
        text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
    return parse_cards(text)
