"""Phase 1.5 — English canonical labels at extraction (prereq for the concept layer).

Acceptance (execution plan §1.5): ≥90 % of nodes carry a plausible `label_en`;
the same concept in two languages yields identical normalized `label_en` in
≥70 % of sampled pairs (the remainder is Phase 6.0's job).
"""

from __future__ import annotations

import pytest

from graphify_ent.labels import (
    ENT_EXTRACTION_SYSTEM,
    coverage_report,
    detect_lang,
    enrich_nodes,
    normalize_label_en,
)


class TestNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("The Béchamel Sauces", "bechamel sauce"),
            ("  EGGPLANTS  ", "eggplant"),
            ("a Roux", "roux"),
            ("Les Tomates", "tomate"),
            ("Knives", "knife"),
            ("Dishes", "dish"),
            ("Potatoes", "potato"),
            ("stock", "stock"),
        ],
    )
    def test_rules_lowercase_singular_no_articles(self, raw, expected):
        assert normalize_label_en(raw) == expected

    def test_idempotent(self):
        once = normalize_label_en("The Sauces")
        assert normalize_label_en(once) == once

    def test_irregular_plurals_are_not_mangled(self):
        # Words ending in -s that are not plurals must survive.
        for w in ("bouillabaisse", "mise en place", "couscous"):
            assert normalize_label_en(w) == w

    def test_empty_and_none_safe(self):
        assert normalize_label_en("") == ""
        assert normalize_label_en(None) == ""


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "text,lang",
        [
            ("la sauce béchamel est préparée avec du beurre", "fr"),
            ("the sauce is prepared with butter and flour", "en"),
            ("la salsa besciamella è preparata con burro", "it"),
        ],
    )
    def test_detects_corpus_languages(self, text, lang):
        assert detect_lang(text) == lang

    def test_unknown_defaults_to_none(self):
        assert detect_lang("") is None


class TestPromptContract:
    def test_prompt_requests_the_three_fields(self):
        for field in ("label_en", "label_orig", "lang"):
            assert field in ENT_EXTRACTION_SYSTEM, f"prompt must request {field}"

    def test_prompt_states_normalization_rules(self):
        low = ENT_EXTRACTION_SYSTEM.lower()
        assert "lowercase" in low and "singular" in low and "article" in low

    def test_prompt_preserves_upstream_confidence_taxonomy(self):
        for tag in ("EXTRACTED", "INFERRED", "AMBIGUOUS"):
            assert tag in ENT_EXTRACTION_SYSTEM

    def test_prompt_preserves_evidence_binding_discipline(self):
        assert "evidence" in ENT_EXTRACTION_SYSTEM.lower()


class TestEnrichment:
    def test_fills_label_en_when_model_omits_it(self):
        nodes = [{"id": "n1", "label": "Les Tomates"}]
        out = enrich_nodes(nodes, default_lang="fr")
        assert out[0]["label_orig"] == "Les Tomates"
        assert out[0]["label_en"] == "tomate"
        assert out[0]["lang"] == "fr"

    def test_respects_model_supplied_label_en(self):
        nodes = [{"id": "n1", "label": "Les Tomates", "label_en": "Tomatoes", "lang": "fr"}]
        out = enrich_nodes(nodes, default_lang=None)
        assert out[0]["label_en"] == "tomato", "model value is normalized, not discarded"

    def test_never_drops_nodes(self):
        nodes = [{"id": f"n{i}"} for i in range(5)]
        assert len(enrich_nodes(nodes, default_lang="en")) == 5

    def test_coverage_report_measures_acceptance(self):
        nodes = enrich_nodes([{"id": "a", "label": "Sauces"}, {"id": "b"}], default_lang="en")
        rep = coverage_report(nodes)
        assert rep["total"] == 2
        assert rep["with_label_en"] == 1
        assert rep["coverage_pct"] == 50.0

    def test_cross_language_pair_converges(self):
        """Same concept, two languages → identical normalized label_en."""
        en = enrich_nodes([{"id": "a", "label": "Eggplants", "label_en": "Eggplants"}], "en")
        it = enrich_nodes([{"id": "b", "label": "Melanzane", "label_en": "Eggplant"}], "it")
        assert en[0]["label_en"] == it[0]["label_en"] == "eggplant"
