"""Adjudication tests for graphify_ent.verify.

Each case here is a mistake the verifier actually made against the real corpus
before it was fixed, so they are regressions, not hypotheticals.
"""

from __future__ import annotations

import pytest

from graphify_ent.verify import (
    CONTRADICTED,
    NOT_FOUND,
    SUPPORTED,
    Claim,
    Verifier,
    quantities,
)

MORNAY = (
    "Cheddar Cheese Sauce: Add 1 lb/454 g grated sharp Cheddar. "
    "Mornay Sauce: Add 8 oz/227 g each grated Gruyère and Parmesan. "
    "Finish with up to 8 oz/227 g whole butter, if desired."
)
BECHAMEL = (
    "Béchamel Sauce Makes 1 gal/3.84 L Ingredients Clarified butter 2 tbsp 30 mL "
    "Minced onion 2 oz 57 g White Roux 1 lb 454 g Milk 5 qt 4.80 L Grated nutmeg"
)
HOLLANDAISE = (
    "Hollandaise Sauce Makes 28 fl oz/840 mL Ingredients Cider vinegar 3 fl oz 90 mL "
    "Egg yolks, fresh or pasteurized 6 fl oz 180 mL Melted butter 18 fl oz 540 mL"
)
MAYONNAISE = (
    "Quality Criteria of Mayonnaise. A properly prepared mayonnaise will be white, "
    "thick and creamy. Elsewhere on this page a roux is mentioned in passing."
)


class _Retriever:
    """Serves fixed passages; the adjudication is what is under test."""

    def __init__(self, docs):
        self.docs = docs

    def query(self, *a, **kw):
        class R:
            refused = False
            hits = [type("H", (), {"node_id": str(i)})() for i in range(len(self.docs))]
            channel_counts = {"fast_path": 1}
        return R()

    def hydrate(self, ids):
        return {str(i): d for i, d in enumerate(self.docs)}


def _doc(passage, page=1):
    return {"passage": passage, "source_file": "book.pdf",
            "source_location": f"pages {page}-{page}", "text_excerpt": passage[:200]}


def _verifier(*passages):
    return Verifier(_Retriever([_doc(p, i + 1) for i, p in enumerate(passages)]))


class TestQuantities:
    def test_normalises_to_one_unit(self):
        assert quantities("8 oz") == pytest.approx([(226.796, "g")], rel=1e-3)
        assert quantities("4.80 L") == [(4800.0, "ml")]

    def test_reads_several_from_one_line(self):
        assert len(quantities("1 lb/454 g grated cheese and 2 tbsp oil")) == 3


class TestSubjectGate:
    def test_a_passage_about_another_sauce_cannot_settle_the_claim(self):
        """Hollandaise names egg yolks. That says nothing about Mornay, and
        accepting it was the bug that confirmed every claim on the card."""
        v = _verifier(HOLLANDAISE)
        f = v.check(Claim("Mornay Sauce", "egg yolks"))
        assert f.verdict == NOT_FOUND

    def test_scattered_subject_words_do_not_count_as_naming_it(self):
        """"white" and "roux" both appear on a page about mayonnaise."""
        v = _verifier(MAYONNAISE)
        assert v.check(Claim("white roux", "cooked without coloring")).verdict == NOT_FOUND

    def test_a_table_row_still_names_the_subject(self):
        """The source lists derivatives as "Mornay  Gruyère and Parmesan" —
        the exact phrase "Mornay Sauce" never occurs, and it must still match."""
        v = _verifier("Name of Derivative. Mornay Gruyère and Parmesan. "
                      "Finish with butter. Poached fish")
        assert v.check(Claim("Mornay Sauce", "Gruyere Parmesan")).verdict == SUPPORTED


class TestAccents:
    def test_unaccented_query_matches_accented_source(self):
        """People type "bechamel" and "gruyere"; the book writes them with
        accents. The index folds, so the adjudication must fold too."""
        v = _verifier(BECHAMEL)
        assert v.check(Claim("Bechamel Sauce", "white roux", "454 g")).verdict == SUPPORTED

    def test_gruyere_without_accent(self):
        v = _verifier(MORNAY)
        assert v.check(Claim("Mornay Sauce", "Gruyere Parmesan")).verdict == SUPPORTED

    def test_soft_hyphen_inside_the_subject_is_invisible(self):
        """PDF extraction writes "two-\xadstage cooling" with a soft hyphen.
        Measured on the CIA book: retrieval returned the right page and the
        adjudication rejected it on a character no reader can see."""
        v = _verifier("In the two-\xadstage cooling method, foods must be "
                      "cooled to 70°F/21°C within 2 hours.")
        assert v.check(
            Claim("two-stage cooling", "cooled to 21°C within 2 hours")
        ).verdict == SUPPORTED


class TestVerdicts:
    def test_supported_carries_the_deciding_sentence_not_the_page_head(self):
        v = _verifier(MORNAY)
        f = v.check(Claim("Mornay Sauce", "grated Gruyere and Parmesan", "227 g"))
        assert f.verdict == SUPPORTED
        assert "227 g each grated" in f.evidence

    def test_absent_ingredient_is_not_found_not_supported(self):
        v = _verifier(MORNAY)
        assert v.check(Claim("Mornay Sauce", "Grana Padano")).verdict == NOT_FOUND

    def test_a_different_figure_for_the_same_thing_is_a_contradiction(self):
        v = _verifier(BECHAMEL)
        f = v.check(Claim("Bechamel Sauce", "white roux", "80 g"))
        assert f.verdict == CONTRADICTED
        assert "differs" in f.detail

    def test_rounding_is_not_a_contradiction(self):
        """"1 lb" and "454 g" are the same quantity written twice."""
        v = _verifier(BECHAMEL)
        assert v.check(Claim("Bechamel Sauce", "white roux", "450 g")).verdict == SUPPORTED

    def test_empty_corpus_is_not_found(self):
        v = _verifier()
        assert v.check(Claim("Mornay Sauce", "anything")).verdict == NOT_FOUND

    def test_findings_are_serialisable(self):
        v = _verifier(MORNAY)
        d = v.check(Claim("Mornay Sauce", "Gruyere Parmesan")).as_dict()
        assert d["verdict"] == SUPPORTED and d["source_location"]


# ---------------------------------------------------------------- scoping
# Where a figure is read from decides both halves of Q1: read too widely and a
# neighbour's number confirms a false claim; read nothing and a true one goes
# unverified. Every case below was measured on the card set (evidence/T99).

DONENESS_TABLE = """Temperatures and Descriptions of Degrees
of Doneness
Meats are cooked to the internal temperature shown below.
Degree of Doneness
Final Resting
Temperature
Description
Fresh beef, veal, and lamb
Rare
135°F/57°C
Interior appearance shiny
Medium
160°F/71°C
Pink to light pink
Fresh pork
Medium
160°F/71°C
Meat opaque throughout
Poultry
Whole birds (chicken, turkey, duck, goose)
180°F/82°C
Leg easy to move in socket
Ground meat and meat mixtures
Turkey, chicken
165°F/74°C
Opaque throughout; juices clear
Beef, veal, lamb, pork
160°F/71°C
Opaque, may have blush of red
Seafood
Fish
145°F/63°C
Still moist"""

REHEAT_PROSE = (
    "Reheat Foods Safely\n"
    "When foods are prepared ahead and then reheated, they should move through "
    "the danger zone as rapidly as possible and be reheated to at least "
    "165°F/74°C for a minimum of 15 seconds. If all proper cooling and "
    "reheating procedures are followed each time, foods may be cooled and "
    "reheated more than once. Do not use hot-holding equipment for reheating. "
    "A steam table will adequately hold reheated foods above 135°F/57°C, but "
    "it will not bring foods out of the danger zone quickly enough."
)


class TestTabularDetection:
    def test_a_doneness_table_is_tabular(self):
        assert Verifier(None)._is_tabular(DONENESS_TABLE)

    def test_prose_with_many_temperatures_is_not(self):
        """Counting figures alone called this page a table, and the row logic
        then read two lines past a sentence into the next paragraph."""
        assert not Verifier(None)._is_tabular(REHEAT_PROSE)

    def test_a_page_with_few_figures_is_not(self):
        assert not Verifier(None)._is_tabular(BECHAMEL)


class TestNestedTableRows:
    """The table is nested: a family header governs several rows, and no single
    word of the subject picks the right one. Anchoring on the longest word read
    the poultry row; anchoring on the rarest read the header, which is the same
    row again. Both confirmed 74 °C for ground beef."""

    def _scope(self, subject, aspect=""):
        return Verifier(None)._scopes(Claim(subject, aspect, "1 g"), DONENESS_TABLE)[0]

    def test_ground_beef_reads_the_beef_row(self):
        scope = self._scope("ground beef", "cooked to internal temperature of")
        assert "Beef, veal, lamb, pork" in scope
        assert "71" in scope and "74" not in scope

    def test_ground_turkey_reads_the_poultry_row(self):
        scope = self._scope("ground turkey", "cooked to internal temperature of")
        assert "Turkey, chicken" in scope and "74" in scope

    def test_a_row_that_says_nothing_of_the_subject_is_not_a_candidate(self):
        assert "Fish" not in self._scope("ground beef")

    def test_the_true_figure_is_confirmed(self):
        ok, why = Verifier(None)._match(
            Claim("ground beef", "cooked to internal temperature of", "71°C"),
            DONENESS_TABLE)
        assert ok, why

    def test_the_neighbouring_rows_figure_is_a_contradiction(self):
        """The Q1 case: 74 °C is the ground-POULTRY line. Confirming it for
        beef is manufactured evidence."""
        ok, why = Verifier(None)._match(
            Claim("ground beef", "cooked to internal temperature of", "74°C"),
            DONENESS_TABLE)
        assert not ok
        assert why.startswith("differs"), why


class TestProseScopes:
    def test_the_aspect_picks_the_sentence(self):
        """Two sentences of one page speak about reheating and give different
        figures: 74 °C "to at least", 57 °C on a steam table. The subject alone
        holds both — and calling that ambiguous is correct and useless."""
        ok, why = Verifier(None)._match(
            Claim("reheated foods", "reheated to at least", "74°C"), REHEAT_PROSE)
        assert ok, why

    def test_the_other_rule_of_the_same_page_still_reads_its_own_figure(self):
        ok, why = Verifier(None)._match(
            Claim("reheated foods", "held on a steam table above", "57°C"), REHEAT_PROSE)
        assert ok, why

    def test_a_figure_the_page_does_not_give_is_a_contradiction(self):
        ok, why = Verifier(None)._match(
            Claim("reheated foods", "reheated to at least", "63°C"), REHEAT_PROSE)
        assert not ok and why.startswith("differs"), why

    def test_scopes_are_ordered_tightest_first(self):
        scopes = Verifier(None)._scopes(
            Claim("reheated foods", "reheated to at least", "74°C"), REHEAT_PROSE)
        assert len(scopes) >= 2
        assert len(scopes[0]) <= len(scopes[-1])

    def test_ambiguity_is_still_declared_when_the_scope_really_holds_two(self):
        """The guard is not weakened, only aimed: a sentence that genuinely
        gives two temperatures for one subject is ambiguous, and saying so is
        the correct answer."""
        text = ("Chill the custard: bring it to 84°C, then hold it at 60°C "
                "before service.")
        ok, why = Verifier(None)._match(Claim("custard", "bring it to", "84°C"), text)
        assert not ok
        assert "ambiguous" in why, why
