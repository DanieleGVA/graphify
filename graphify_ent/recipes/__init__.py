"""The recipe layer: a recipe recognised for what it is, not what it is called.

Direction fixed by the T96 experiment and validated before any of this was
built: a recipe's identity is its **ingredient fingerprint** (canonical
ingredients and their proportions, weighted by rarity, scale-invariant), then
its **procedure**, and only last its **title** — a Tiramisù given a fantasy name
still ranked first against 1,578 pages.

Nothing here calls a model. A wrong answer must be explainable by reading the
text it came from.
"""

from graphify_ent.recipes.ingredients import (
    CONVERTED_PIECE,
    CONVERTED_VOLUME,
    MEASURED,
    UNQUANTIFIED,
    Registry,
    Resolved,
    parse_block,
    proportions,
    resolve_line,
)

__all__ = [
    "CONVERTED_PIECE",
    "CONVERTED_VOLUME",
    "MEASURED",
    "UNQUANTIFIED",
    "Registry",
    "Resolved",
    "parse_block",
    "proportions",
    "resolve_line",
]
