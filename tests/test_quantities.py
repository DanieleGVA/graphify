"""The one quantity grammar (ADR-0004 Q2).

Every case here either produced a wrong verdict against the real corpus
(evidence/verify-canonical-prototype/) or was a measured failure of an earlier
parser, so these are regressions, not hypotheticals. The unit values asserted
are the YAML's — the single surviving table — not the culinary-exact ones the
retired verify table carried.
"""

from __future__ import annotations

import pytest

from graphify_ent.quantities import (
    UNCERTAINTY,
    MEASURED,
    UNQUANTIFIED,
    UnitTable,
    default_table,
    norm,
    number,
    scan,
)


@pytest.fixture(scope="module")
def table() -> UnitTable:
    return default_table()


def figures(text, table):
    return [(round(q.lo, 1), round(q.hi, 1), q.unit) for q in scan(text, table)]


class TestNumbers:
    def test_mixed_number_is_not_a_bare_integer(self):
        assert number("1 1/2") == 1.5

    def test_ascii_fraction(self):
        assert number("1/2") == 0.5

    def test_comma_decimal(self):
        assert number("4,8") == 4.8

    def test_vulgar_fraction_with_whole(self):
        assert number("1½") == 1.5


class TestScan:
    def test_the_defect_unit_parses(self, table):
        """"1 gal" absent from every table was the verify_claim defect: a
        figure no table knows is a figure nothing can contradict."""
        assert figures("1 gal", table) == [(3785.0, 3785.0, "ml")]

    def test_dual_notation_is_two_statements_of_one_figure(self, table):
        assert figures("5 qt/4.80 L", table) == [
            (4730.0, 4730.0, "ml"), (4800.0, 4800.0, "ml")]

    def test_the_fraction_misparse_never_returns(self, table):
        """The lexical patch read "1/2 cup" as 2 cups (473 ml for 120): the
        denominator sat next to the unit. The mixed/fraction alternation must
        always win over the bare integer."""
        assert figures("1/2 cup", table) == [(120.0, 120.0, "ml")]

    def test_the_nfkd_fraction_slash_trap(self, table):
        """NFKD decomposes "½" into "1⁄2" with a FRACTION SLASH; unmapped, the
        "2" was read against the unit. Measured on R1 before it was fixed."""
        assert figures("½ cup", table) == [(120.0, 120.0, "ml")]

    def test_a_range_is_one_figure_with_two_endpoints(self, table):
        got = scan("2 to 3 qt", table)
        assert len(got) == 1 and (got[0].lo, got[0].hi) == (1892.0, 2838.0)

    def test_an_italian_range(self, table):
        got = scan("da 165 a 180 °C", table)
        assert len(got) == 1 and got[0].unit == "c"
        assert (got[0].lo, got[0].hi) == (165.0, 180.0)

    def test_fahrenheit_converts_with_offset_not_a_factor(self, table):
        got = scan("165°F/74°C", table)
        assert [q.unit for q in got] == ["c", "c"]
        assert round(got[0].lo, 1) == 73.9 and got[1].lo == 74.0

    def test_compound_us_mass(self, table):
        assert figures("1 lb 2 oz", table) == [
            (453.6, 453.6, "g"), (56.7, 56.7, "g")]

    def test_nonbreaking_space_from_pdf_extraction(self, table):
        assert figures("Makes 1\xa0gal/3.84\xa0L", table) == [
            (3785.0, 3785.0, "ml"), (3840.0, 3840.0, "ml")]

    def test_what_is_not_a_quantity_yields_nothing(self, table):
        """Absence is the refusal signal downstream (ADR-0004 Q4): a page
        reference or a count must never come back wearing a unit."""
        for text in ("page 253", "3 eggs", "30 minutes", "a pinch", ""):
            assert scan(text, table) == []


class TestUnitTable:
    def test_the_yaml_is_the_single_surviving_table(self, table):
        """gal was missing and cup/qt were double-defined in code (ADR-0004
        'data gaps'); they must all resolve from the YAML now."""
        assert table.to_base(1, "gal") == (3785.0, "ml")
        assert table.to_base(1, "cup") == (240.0, "ml")
        assert table.to_base(1, "qt") == (946.0, "ml")

    def test_temperature_scales(self, table):
        assert table.to_base(212, "°F") == (100.0, "c")
        assert table.to_base(74, "°c") == (74.0, "c")

    def test_an_unknown_unit_raises_rather_than_guesses(self, table):
        with pytest.raises(KeyError):
            table.to_base(1, "cubit")

    def test_an_empty_table_matches_nothing(self):
        assert scan("5 qt of anything", UnitTable()) == []


class TestPromotion:
    def test_recipe_layer_reexports_the_same_objects(self):
        """R1's public API is unchanged by the promotion: same objects, not
        copies that could drift."""
        from graphify_ent.recipes import ingredients as ing
        import graphify_ent.quantities as q

        assert ing.MEASURED is q.MEASURED is MEASURED
        assert ing.UNQUANTIFIED is q.UNQUANTIFIED is UNQUANTIFIED
        assert ing.norm is q.norm is norm
        assert ing.UNCERTAINTY is q.UNCERTAINTY is UNCERTAINTY

    def test_registry_reads_the_new_units_too(self):
        """The registry builds its pattern from the same YAML, so the closed
        data gap reaches R1 as well: a '1 gal' line is now a volume."""
        from graphify_ent.recipes.ingredients import Registry

        reg = Registry.load()
        assert ("1 gal", [(1.0, "gal", "volume")]) == (
            "1 gal", reg.quantities("cream 1 gal"))
