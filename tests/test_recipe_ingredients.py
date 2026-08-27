"""R1 — canonical ingredients and the gravimetric standard.

Every case here is a mistake the parser actually made against the real corpus
before it was fixed (evidence/T96, T98), so they are regressions rather than
hypotheticals. The two golden fixtures are verbatim text: the Mascarpone
Filling table of Professional Baking p. 489 as the graph holds it, and an MSC
recipe card in the shape that broke the first version.
"""

from __future__ import annotations

import textwrap

import pytest

from graphify_ent.recipes import (
    CONVERTED_PIECE,
    CONVERTED_VOLUME,
    MEASURED,
    UNQUANTIFIED,
    Registry,
    parse_block,
    proportions,
    resolve_line,
)


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load()


# Professional Baking (Gisslen) p. 489, Mascarpone Filling — the two-column
# US/Metric table as PDF extraction flattens it: the name, then one line per
# column. Nothing is edited.
GISSLEN_489 = textwrap.dedent("""\
    Egg yolks
    2 yolks
    2 yolks
    Sugar
    6 oz
    180 g
    Water
    4 oz
    120 g
    Glucose or corn syrup
    2 oz
    60 g
    Mascarpone
    1 lb
    500 g
    Heavy cream
    1 lb   8 oz
    740 g
    """)

# An MSC card in the shape that produced the two measured failures: a section
# header above the first ingredient, and trade names carrying their own figures.
MSC_CARD = textwrap.dedent("""\
    COFFEE SYRUP
    WATER 2000 g
    SUGAR 400 g
    COFFEE INSTANT 200G NESCAFE 60 g
    CREAM WHIPPING 36% UHT SUGAR FREE 1500 g
    """)


class TestGoldenFixtures:
    def test_gisslen_489_parses_exactly(self, reg):
        got = {r.canonical: r.grams for r in parse_block(GISSLEN_489, reg)}
        assert got == {"egg_yolk": 36.0, "sugar": 180.0, "water": 120.0,
                       "glucose": 60.0, "mascarpone": 500.0, "cream": 740.0}

    def test_gisslen_489_classes_are_declared(self, reg):
        got = {r.canonical: r.cls for r in parse_block(GISSLEN_489, reg)}
        assert got["egg_yolk"] == CONVERTED_PIECE, "counted, not weighed"
        assert got["mascarpone"] == MEASURED

    def test_msc_card_parses_exactly(self, reg):
        got = {(r.canonical, r.cls): r.grams for r in parse_block(MSC_CARD, reg)}
        assert got[("water", MEASURED)] == 2000.0
        assert got[("sugar", MEASURED)] == 400.0
        assert got[("coffee", MEASURED)] == 60.0
        assert got[("cream", MEASURED)] == 1500.0


class TestHeadersNeverAbsorbQuantities:
    """The measured failure: "COFFEE SYRUP" took the 2,000 g of the water line
    beneath it AND the water kept it, so the query became a sweet coffee liquid
    whose best match in 1,743 pages was a BBQ sauce."""

    def test_the_header_gets_nothing(self, reg):
        rows = parse_block(MSC_CARD, reg)
        header = [r for r in rows if r.canonical == "coffee" and r.raw.startswith("COFFEE SYRUP")]
        assert header and header[0].grams is None
        assert header[0].cls == UNQUANTIFIED

    def test_the_water_keeps_its_own_figure(self, reg):
        water = [r for r in parse_block(MSC_CARD, reg) if r.canonical == "water"]
        assert len(water) == 1 and water[0].grams == 2000.0

    def test_a_quantity_is_never_counted_twice(self, reg):
        rows = parse_block(MSC_CARD, reg)
        grams = [r.grams for r in rows if r.quantified]
        assert sum(grams) == 2000.0 + 400.0 + 60.0 + 1500.0

    def test_a_header_still_counts_as_present(self, reg):
        """Presence is evidence — rarity weighting makes it half of identity —
        so a header is kept without a figure, never dropped."""
        canonicals = {r.canonical for r in parse_block(MSC_CARD, reg)}
        assert "coffee" in canonicals


class TestTradeNamesDoNotReclassify:
    """Two failures on one card: "CREAM WHIPPING 36% UHT SUGAR FREE" was read
    as sugar and the cream vanished; "COFFEE INSTANT 200G NESCAFE" gave up its
    pack size instead of its recipe quantity."""

    def test_sugar_free_cream_is_cream(self, reg):
        r = resolve_line("CREAM WHIPPING 36% UHT SUGAR FREE 1500 g", reg)
        assert r.canonical == "cream" and r.grams == 1500.0

    def test_the_pack_size_is_not_the_quantity(self, reg):
        r = resolve_line("COFFEE INSTANT 200G NESCAFE 60 g", reg)
        assert r.canonical == "coffee" and r.grams == 60.0

    def test_the_leftmost_synonym_decides(self, reg):
        assert reg.canonical("BUTTER UNSALTED SUGAR FREE") == "butter"
        assert reg.canonical("SUGAR ICING") == "powdered_sugar"

    def test_a_longer_synonym_wins_at_the_same_position(self, reg):
        assert reg.canonical("egg whites 8 oz") == "egg_white"
        assert reg.canonical("eggs 4") == "egg"


class TestUsMetricTwoColumn:
    def test_the_metric_column_is_taken(self, reg):
        assert resolve_line("Sugar 6 oz 180 g", reg).grams == 180.0

    def test_us_only_compounds_instead_of_competing(self, reg):
        """"1 lb 8 oz" is one weight written in two units."""
        r = resolve_line("Heavy cream 1 lb 8 oz", reg)
        assert r.grams == pytest.approx(453.6 + 8 * 28.35, rel=1e-6)

    def test_the_metric_column_survives_a_trade_name_figure(self, reg):
        assert resolve_line("COFFEE INSTANT 200G NESCAFE 60 g", reg).grams == 60.0


class TestReflowedLines:
    """epub text arrives with rows broken across lines and no table structure.
    The row must close on the next INGREDIENT, not on the next line break."""

    def test_a_row_broken_across_lines_is_one_row(self, reg):
        text = "Mascarpone\n500\ng\nSugar\n180 g\n"
        got = {r.canonical: r.grams for r in parse_block(text, reg)}
        assert got == {"mascarpone": 500.0, "sugar": 180.0}

    def test_a_repeated_name_across_columns_is_not_two_ingredients(self, reg):
        """"Egg yolks / 2 yolks / 2 yolks" is one row printed twice, and
        reading it as two doubled the recipe."""
        rows = parse_block("Egg yolks\n2 yolks\n2 yolks\n", reg)
        assert len(rows) == 1 and rows[0].grams == 36.0

    def test_blank_lines_do_not_close_a_row(self, reg):
        rows = parse_block("Mascarpone\n\n500 g\n", reg)
        assert len(rows) == 1 and rows[0].grams == 500.0


class TestGravimetry:
    def test_a_stated_mass_is_exact(self, reg):
        r = resolve_line("Mascarpone 500 g", reg)
        assert (r.cls, r.uncertainty) == (MEASURED, 0.0)

    def test_a_volume_uses_the_ingredients_density(self, reg):
        """A cup is not a weight: it is 120 g of flour and 204 g of sugar."""
        assert resolve_line("1 cup all-purpose flour", reg).grams == pytest.approx(120.0)
        assert resolve_line("1 cup sugar", reg).grams == pytest.approx(204.0)
        assert resolve_line("1 cup honey", reg).grams == pytest.approx(340.8)

    def test_state_is_part_of_the_ingredient(self, reg):
        """Whipping halves the density. Taking one entry for the other was a
        150% error when it was measured (T98), which is why the state is in
        the name rather than in a comment."""
        plain = resolve_line("1 cup cream", reg).grams
        whipped = resolve_line("1 cup whipped cream", reg).grams
        assert plain > whipped * 2

    def test_a_count_uses_the_piece_weight(self, reg):
        r = resolve_line("3 egg yolks", reg)
        assert (r.grams, r.cls, r.uncertainty) == (54.0, CONVERTED_PIECE, 0.10)

    def test_conversions_declare_their_uncertainty(self, reg):
        assert resolve_line("1 cup flour", reg).uncertainty == 0.15
        assert resolve_line("120 g flour", reg).uncertainty == 0.0

    def test_to_taste_is_present_but_unweighed(self, reg):
        r = resolve_line("salt, to taste", reg)
        assert r.canonical == "salt" and r.grams is None and r.cls == UNQUANTIFIED

    def test_a_volume_without_a_density_is_declared_not_guessed(self, reg):
        """A missing density is not a silent 1.0 g/ml: inventing a substance is
        exactly the kind of quiet conversion the classes exist to prevent."""
        data = {"version": 1,
                "units": {"volume_ml": {"cup": 240.0}, "mass_g": {"g": 1.0}},
                "ingredients": {"mystery": {"synonyms": ["mystery"]}}}
        r = resolve_line("2 cups mystery", Registry(data))
        assert r.canonical == "mystery" and r.grams is None
        assert r.cls == UNQUANTIFIED

    def test_fractions_and_unicode_fractions(self, reg):
        assert resolve_line("½ cup water", reg).grams == pytest.approx(120.0)
        assert resolve_line("1 ½ cups water", reg).grams == pytest.approx(360.0)
        assert resolve_line("1/2 cup water", reg).grams == pytest.approx(120.0)

    def test_a_comma_decimal_is_a_decimal(self, reg):
        """Three languages, two decimal marks."""
        assert resolve_line("farina 1,5 kg", reg).grams == pytest.approx(1500.0)


class TestProportions:
    def test_proportions_sum_to_one(self, reg):
        p = proportions(parse_block(GISSLEN_489, reg))
        assert sum(p.values()) == pytest.approx(1.0)

    def test_scale_invariance(self, reg):
        """The property the whole matching idea rests on: a recipe for ten and
        the same recipe for a hundred are the same recipe."""
        single = proportions(parse_block("Flour 500 g\nWater 300 g\n", reg))
        tenfold = proportions(parse_block("Flour 5 kg\nWater 3 kg\n", reg))
        assert single == pytest.approx(tenfold)

    def test_unquantified_ingredients_are_absent_from_proportions(self, reg):
        p = proportions(parse_block("Flour 500 g\nsalt, to taste\n", reg))
        assert set(p) == {"flour"}

    def test_an_all_unquantified_block_has_no_proportions(self, reg):
        assert proportions(parse_block("salt, to taste\npepper, to taste\n", reg)) == {}

    def test_repeated_ingredients_add_up(self, reg):
        p = proportions(parse_block("Butter 100 g\nSugar 100 g\nButter 200 g\n", reg))
        assert p["butter"] == pytest.approx(0.75)


class TestRegistryIsPluggable:
    """DOMAIN-AGNOSTIC rule: nothing in the module may know an ingredient by
    name. The vocabulary is a file, and another domain replaces it wholesale."""

    def test_an_empty_registry_recognises_nothing(self):
        empty = Registry({"version": 1, "units": {}, "ingredients": {}})
        assert empty.canonical("Mascarpone 500 g") is None
        assert resolve_line("Mascarpone 500 g", empty) is None

    def test_a_foreign_registry_works(self):
        data = {"version": 1,
                "units": {"volume_ml": {"l": 1000.0}, "mass_g": {"g": 1.0, "kg": 1000.0}},
                "ingredients": {"resin": {"synonyms": ["epoxy resin", "resin"],
                                          "density": 1.1}}}
        r = resolve_line("Epoxy resin 2 kg", Registry(data))
        assert (r.canonical, r.grams) == ("resin", 2000.0)

    def test_the_shipped_registry_declares_its_version(self, reg):
        assert reg.version >= 1
        assert len(reg.ingredients) >= 60

    def test_no_ingredient_name_is_hardcoded_in_the_module(self):
        """The check that keeps the rule honest as the module grows — and it
        follows the grammar to its promoted home (ADR-0004 Q2)."""
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent / "graphify_ent"
        # file -> first line of code after its module docstring
        anchors = {root / "recipes" / "ingredients.py": "DEFAULT_REGISTRY = ",
                   root / "quantities.py": "MEASURED = "}
        for path, anchor in anchors.items():
            src = path.read_text()
            code = "\n".join(ln for ln in src.splitlines()
                             if not ln.lstrip().startswith("#"))
            code = code[code.index(anchor):]           # past the module docstring
            for name in ("mascarpone", "ladyfinger", "savoiardi", "gruyere"):
                assert name not in code.lower(), f"{name} in {path.name}"
