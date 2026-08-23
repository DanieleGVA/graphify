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
