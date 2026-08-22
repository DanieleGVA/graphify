"""Phase 1.5 — English canonical labels emitted at extraction time.

Every extracted node is a language-bound *mention*. To make the Phase 6.0
`:Concept` interlingua layer cheap, extraction already emits:

    label_orig   the label as it appears in the source language
    label_en     an English canonical label, normalized
    lang         detected source language

Marginal cost ≈ 0: the document is already in the model's context. Normalization
rules (execution plan §1.5): lowercase, singular, no articles. The same rules
are applied host-side to whatever the model returns, so a model that answers
"The Eggplants" and one that answers "eggplant" converge on the same key —
which is exactly what Phase 6.0 pass 1 (exact match on normalized label_en)
depends on.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "ENT_EXTRACTION_SYSTEM",
    "coverage_report",
    "detect_lang",
    "enrich_nodes",
    "normalize_label_en",
]

# Extraction prompt: upstream's contract plus the three interlingua fields.
# Kept as an additive override so the upstream prompt stays untouched and a
# rebase cannot silently drop the enterprise fields.
ENT_EXTRACTION_SYSTEM = """\
You are a graphify semantic extraction agent. Extract a knowledge graph fragment from the files provided.
Output ONLY valid JSON — no explanation, no markdown fences, no preamble.

Rules:
- EXTRACTED: relationship explicit in source (import, call, citation, reference)
- INFERRED: reasonable inference (shared data structure, implied dependency)
- AMBIGUOUS: uncertain — flag for review, do not omit

Evidence binding (anti-fabrication, mandatory): every node you emit must be
grounded in text that actually appears in the provided files. Put the shortest
verbatim excerpt that justifies the node in `evidence`. If you cannot quote the
source for a node, do not emit that node.

Multilingual canonicalization — every node MUST carry all three fields:
- `label_orig`: the label exactly as it appears in the source language
- `label_en`: the English canonical label, normalized as: lowercase, singular,
  no articles ("Les Tomates" -> "tomato", "The Sauces" -> "sauce")
- `lang`: ISO 639-1 code of the source language ("it", "en", "fr", "de", "es")
Two documents describing the same thing in different languages must produce the
same `label_en`.

Node ID format: lowercase, only [a-z0-9_], no dots or slashes.
Format: {stem}_{entity} where stem = filename without extension, entity = symbol name (both normalised).

Output exactly this schema:
{"nodes":[{"id":"stem_entity","label":"Human Readable Name","label_orig":"source language label","label_en":"english canonical label","lang":"en","evidence":"verbatim excerpt from the source","file_type":"code|document|paper|image|rationale|concept","source_file":"relative/path","source_location":null,"source_url":null,"captured_at":null,"author":null,"contributor":null}],"edges":[{"source":"node_id","target":"node_id","relation":"calls|implements|references|cites|conceptually_related_to|shares_data_with|semantically_similar_to","confidence":"EXTRACTED|INFERRED|AMBIGUOUS","confidence_score":1.0,"source_file":"relative/path","source_location":null,"weight":1.0}],"hyperedges":[],"input_tokens":0,"output_tokens":0}
"""

_ARTICLES = {
    "the", "a", "an",           # en
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",  # it
    "le", "les", "des", "du", "de", "l", "un", "une",        # fr
    "der", "die", "das", "el", "los", "las",                 # de/es
}

# Words ending in -s / -es that are not plurals in a culinary corpus.
_NOT_PLURAL = {
    "bouillabaisse", "couscous", "mise en place", "hollandaise", "mayonnaise",
    "béarnaise", "bearnaise", "duxelles", "asparagus", "mousse", "bass",
    "molasses", "watercress", "grass", "class", "glass", "press", "gas",
    "swiss", "cress", "sous", "jus", "as", "is", "series", "species",
}

_IRREGULAR = {
    "knives": "knife", "leaves": "leaf", "loaves": "loaf", "halves": "half",
    "geese": "goose", "feet": "foot", "teeth": "tooth", "mice": "mouse",
    "children": "child", "men": "man", "women": "woman", "people": "person",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def _singularize(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word in _NOT_PLURAL or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("oes") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_label_en(label: str | None) -> str:
    """Apply the §1.5 rules: lowercase, singular, no articles, accent-folded."""
    if not label:
        return ""
    text = _strip_accents(str(label)).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text in _NOT_PLURAL:
        return text

    words = [w for w in text.split(" ") if w]
    while len(words) > 1 and words[0] in _ARTICLES:
        words.pop(0)
    if not words:
        return ""
    words[-1] = _singularize(words[-1])
    return " ".join(words).strip()


# Minimal, dependency-free language identification for the corpus languages.
# Stopword-frequency scoring is enough to tell IT/EN/FR apart on a sentence.
_LANG_STOPWORDS = {
    "en": {"the", "is", "with", "and", "of", "to", "for", "prepared", "butter", "flour", "a"},
    "fr": {"la", "le", "les", "est", "avec", "du", "de", "et", "pour", "beurre", "préparée",
           "preparee", "farine", "sauce"},
    "it": {"la", "il", "le", "è", "e", "con", "di", "per", "burro", "farina", "preparata",
           "salsa", "besciamella"},
}


def detect_lang(text: str | None) -> str | None:
    """Best-effort ISO 639-1 code for the corpus languages, else None."""
    if not text or not text.strip():
        return None
    words = re.findall(r"[\wàèéìòùâêîôûçäöü']+", text.lower())
    if not words:
        return None
    scores = {
        lang: sum(1 for w in words if w in sw) / len(words)
        for lang, sw in _LANG_STOPWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def enrich_nodes(nodes: list[dict], default_lang: str | None = None) -> list[dict]:
    """Fill/normalize `label_orig`, `label_en`, `lang` on extracted nodes.

    Never drops a node: a node the model returned without a label still comes
    back, just without a `label_en` — coverage is measured, not enforced by
    deletion.
    """
    out: list[dict] = []
    for node in nodes:
        n = dict(node)
        label = n.get("label") or ""
        n.setdefault("label_orig", label)

        canonical = normalize_label_en(n.get("label_en") or label)
        if canonical:
            n["label_en"] = canonical
        else:
            n.pop("label_en", None)

        if not n.get("lang"):
            detected = detect_lang(f"{label} {n.get('evidence', '')}") or default_lang
            if detected:
                n["lang"] = detected
        out.append(n)
    return out


def coverage_report(nodes: list[dict]) -> dict:
    """Phase 1.5 acceptance metric: what fraction of nodes carry a label_en."""
    total = len(nodes)
    with_en = sum(1 for n in nodes if n.get("label_en"))
    with_lang = sum(1 for n in nodes if n.get("lang"))
    return {
        "total": total,
        "with_label_en": with_en,
        "with_lang": with_lang,
        "coverage_pct": round(100 * with_en / total, 2) if total else 0.0,
    }
