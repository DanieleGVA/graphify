"""Canonical quantity facts (ADR-0004 Q1-B).

The fixtures are the exact page SHAPES that produced wrong verdicts against
the real corpus (evidence/verify-canonical-prototype/): the flattened recipe
table whose subject line is the yield, the prose block where the yield sits
one sentence away from the milk, and the nested doneness table. Regressions,
not hypotheticals.
"""

from __future__ import annotations

import pytest

from graphify_ent.facts import (
    ANCHOR_UNCERTAINTY,
    PROSE_WINDOW,
    TABLE_ROW,
    Fact,
    derive,
    materialize,
)
from graphify_ent.retrieval import verify_evidence_binding

# The Professional Chef p. 379 as PDF extraction flattens it: each cell its
# own line, the subject line ("Béchamel Sauce") followed by the YIELD — the
# figure that was confirmed as a milk quantity. Non-breaking spaces verbatim.
RECIPE_TABLE = (
    "366  |  MISE EN PLACE, STOCKS, SAUCES, AND SOUPS\n"
    "Béchamel Sauce\n"
    "Makes 1\xa0gal/3.84\xa0L\n"
    "Ingredients\n"
    "Clarified butter or vegetable oil\n"
    "2 tbsp\n"
    "30\xa0mL\n"
    "White Roux (page 253)\n"
    "1\xa0lb\n"
    "454 g\n"
    "Milk\n"
    "5 qt\n"
    "4.80\xa0L\n"
    "Kosher salt\n"
    "As needed\n"
)

# p. 348's appended formula block, one flat line: the yield and the milk
# figure live in the SAME sentence, ~90 characters apart — the measured
# distance at which an unclipped window confused them.
PROSE_BLOCK = (
    "BASIC FORMULA White Sauce (1 gal/3.84 L) Aromatics (white mirepoix, "
    "minced onions), as needed Butter or oil, as needed 5 qt/4.80 L flavorful "
    "liquid (white stock for velouté; milk for béchamel) 1 lb/454 g White or "
    "Blond Roux (page 253)"
)

DONENESS = (
    "Ground meat and meat mixtures\n"
    "Turkey, chicken\n"
    "165°F/74°C\n"
    "Beef, veal, lamb, pork\n"
    "160°F/71°C\n"
    "Seafood\n"
    "Fish\n"
    "145°F/63°C\n"
)


def by_anchor(facts, word):
    return [f for f in facts if word in f.anchor_text.lower()]


class TestTableRows:
    def test_the_milk_row_owns_the_milk_figures(self):
        milk = by_anchor(derive(RECIPE_TABLE), "milk")
        assert sorted(round(f.value_lo) for f in milk) == [4730, 4800]
        assert all(f.anchor_kind == TABLE_ROW for f in milk)

    def test_the_yield_is_not_owned_by_any_ingredient(self):
        """The defect: "1 gal" confirmed as the milk. Its owner is the line
        that says Makes — and only that."""
        gal = [f for f in derive(RECIPE_TABLE) if round(f.value_lo) == 3785]
        assert gal and all("makes" in f.anchor_text.lower() for f in gal)
        assert not any("milk" in f.anchor_text.lower() for f in gal)

    def test_a_row_walks_up_to_the_nearest_worded_line(self):
        roux = by_anchor(derive(RECIPE_TABLE), "roux")
        assert {round(f.value_lo) for f in roux} == {454}
        assert all(f.unit_base == "g" for f in roux)

    def test_nested_doneness_rows_keep_their_own_figures(self):
        facts = derive(DONENESS)
        beef = by_anchor(facts, "beef")
        assert beef and all(round(f.value_lo) == 71 for f in beef)
        turkey = by_anchor(facts, "turkey")
        assert turkey and all(round(f.value_lo) == 74 for f in turkey)


class TestProseWindows:
    def test_the_window_does_not_reach_the_yield(self):
        """±60 clipped at sentence ends — 90 unclipped characters were enough
        to pull the yield into the milk's window (measured)."""
        facts = derive(PROSE_BLOCK)
        milk = by_anchor(facts, "milk")
        assert sorted(round(f.value_lo) for f in milk
                      if f.unit_base == "ml") == [4730, 4800]
        gal = [f for f in facts if round(f.value_lo) == 3785]
        assert gal and not any("milk" in f.anchor_text.lower() for f in gal)

    def test_prose_declares_its_weaker_anchor(self):
        facts = derive(PROSE_BLOCK)
        assert facts and all(f.anchor_kind == PROSE_WINDOW for f in facts)
        assert all(f.uncertainty == ANCHOR_UNCERTAINTY[PROSE_WINDOW]
                   for f in facts)


class TestEvidenceBinding:
    def test_every_raw_text_is_found_in_its_source(self):
        """A quote that cannot be found in its own source must not exist —
        the machine check behind Q1, applied at derivation."""
        for passage in (RECIPE_TABLE, PROSE_BLOCK, DONENESS):
            for f in derive(passage):
                assert verify_evidence_binding(f.raw_text, passage), f.raw_text

    def test_accents_and_nbsp_survive_verbatim(self):
        milk = by_anchor(derive(RECIPE_TABLE), "milk")
        assert any("Milk" in f.raw_text for f in milk)
        gal = [f for f in derive(RECIPE_TABLE) if round(f.value_lo) == 3785]
        assert "1\xa0gal" in gal[0].raw_text

    def test_an_empty_passage_yields_nothing(self):
        assert derive("") == [] and derive("   \n  ") == []


class TestFactIdentity:
    def test_deterministic_and_idempotent(self):
        a, b = derive(RECIPE_TABLE), derive(RECIPE_TABLE)
        assert [f.fact_id("n1", "units-v5") for f in a] == \
               [f.fact_id("n1", "units-v5") for f in b]

    def test_identity_is_per_node_and_per_grammar(self):
        f = derive(RECIPE_TABLE)[0]
        assert f.fact_id("n1", "units-v5") != f.fact_id("n2", "units-v5")
        assert f.fact_id("n1", "units-v5") != f.fact_id("n1", "units-v6")


class _Session:
    """Records what the batch writes; serves one node with a passage."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, **params):
        self.calls.append((query, params))
        if "RETURN n.id" in query:
            return iter(self.nodes)
        if "UNWIND" in query:
            made = len(params["rows"])
            return type("R", (), {"single": staticmethod(lambda: {"made": made})})()
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Loader:
    def __init__(self, nodes):
        self.session = _Session(nodes)

    def _session(self):
        return self.session


class TestMaterialize:
    def test_writes_bound_facts_with_full_schema(self):
        loader = _Loader([{"id": "p379", "passage": RECIPE_TABLE}])
        stats = materialize(loader, "pilot")
        assert stats["nodes"] == 1 and stats["facts"] > 0
        assert stats["created"] == stats["facts"]
        writes = [(q, p) for q, p in loader.session.calls if "UNWIND" in q]
        assert writes, "no write issued"
        q, p = writes[0]
        assert "MERGE (f:QuantityFact {fact_id: r.fact_id})" in q
        assert "ANCHORED_TO" in q and p["domain"] == "pilot"
        # id alone is not unique across domains — the same book loaded twice
        # shares content-derived ids, and an unpinned MATCH anchored 64k
        # facts to the twin domain's nodes (measured, first live run).
        assert "{id: r.node_id, domain: $domain}" in q
        row = p["rows"][0]
        props = row["props"]
        assert {"value_lo", "value_hi", "unit_base", "raw_text",
                "anchor_text", "anchor_kind", "uncertainty"} <= set(props)
        assert p["gv"].startswith("units-v")

    def test_constraint_is_ensured_before_writing(self):
        loader = _Loader([])
        materialize(loader, "pilot")
        assert "CREATE CONSTRAINT quantity_fact_id" in loader.session.calls[0][0]
