"""Settle a documentary claim against the graph — never against the source file.

The motivating case: a kitchen hands over a recipe card that says "reference
followed: <book>", and someone has to decide whether each figure on it actually
matches that book. Doing it by reading the PDF is what the system exists to
replace, so this module talks only to Neo4j. If a fact is not in the graph the
answer is "not found", never "let me open the book" — an unsupported claim is a
finding, not a reason to fall back.

Three verdicts, and the distinction between the last two is the whole point:

  SUPPORTED    the source states this value for this subject
  CONTRADICTED the source states a DIFFERENT value for it — the useful case,
               and the one a similarity search alone will never surface,
               because the contradicting passage is the most similar one
  NOT_FOUND    the corpus does not speak to it. Reported plainly rather than
               resolved by picking the nearest passage.

Every finding carries the passage it was decided from, with book and page, so a
human can overrule it in one glance. Nothing here paraphrases the source.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["Claim", "Finding", "Verifier", "SUPPORTED", "CONTRADICTED", "NOT_FOUND"]

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_FOUND = "NOT_FOUND"

#: Quantities as cookbooks write them: "454 g", "8 oz", "4.80 L", "1 lb 2 oz".
_QTY = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(kg|g|mg|lb|oz|l|litres?|liters?|ml|cl|dl|tbsp|tsp)\b", re.I)

#: Everything normalised to grams or millilitres so "8 oz" and "227 g" compare.
_TO_BASE = {
    "kg": ("g", 1000.0), "g": ("g", 1.0), "mg": ("g", 0.001),
    "lb": ("g", 453.592), "oz": ("g", 28.3495),
    "l": ("ml", 1000.0), "litre": ("ml", 1000.0), "litres": ("ml", 1000.0),
    "liter": ("ml", 1000.0), "liters": ("ml", 1000.0),
    "ml": ("ml", 1.0), "cl": ("ml", 10.0), "dl": ("ml", 100.0),
    "tbsp": ("ml", 15.0), "tsp": ("ml", 5.0),
}


def quantities(text: str) -> list[tuple[float, str]]:
    """Every quantity in `text`, normalised to grams or millilitres."""
    out = []
    for value, unit in _QTY.findall(text or ""):
        base = _TO_BASE.get(unit.lower())
        if not base:
            continue
        out.append((float(value.replace(",", ".")) * base[1], base[0]))
    return out


#: Words that carry no discriminating power in a recipe corpus — requiring
#: them would reject good passages, counting them would inflate coverage.
_WEAK = {"sauce", "cheese", "with", "into", "from", "that", "this", "quantity",
         "amount", "salsa", "formaggio", "and", "the", "for", "grated", "whole",
         "fresh", "made", "make", "used", "using", "serve", "served"}


#: How close the words of a multi-word subject must sit to count as naming it.
_NEAR = 90


#: Invisible characters PDF text extraction leaves inside words: the soft
#: hyphen ("two-\xadstage"), zero-width spaces and the word joiner. Measured on
#: the two-book corpus: the CIA book writes "two-­stage cooling" with a soft
#: hyphen, retrieval returned the right page and the adjudication then rejected
#: it on a character no reader can see.
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠"))


def fold(s: str) -> str:
    """Strip diacritics. The source writes "Béchamel" and "Gruyère"; people
    type "bechamel" and "gruyere". The fulltext index folds — measured, without
    it "gruyere" matched nothing — and the adjudication has to fold too, or a
    passage the search correctly returned is then rejected on the accent."""
    return "".join(c for c in unicodedata.normalize("NFKD", (s or "").translate(_INVISIBLE))
                   if not unicodedata.combining(c))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", fold(s)).strip().lower()


@dataclass
class Claim:
    """One assertion to check.

    `subject` is what the claim is about and drives retrieval; `aspect` narrows
    it (an ingredient, a step); `value` is the figure asserted, if any. A claim
    with no value is checked for presence only — which is how "the recipe
    contains egg yolks" gets answered.
    """
    subject: str
    aspect: str = ""
    value: str = ""
    note: str = ""

    @property
    def query(self) -> str:
        return f"{self.subject} {self.aspect}".strip()


@dataclass
class Finding:
    claim: Claim
    verdict: str
    evidence: str = ""
    source_file: str = ""
    source_location: str = ""
    detail: str = ""
    latency_ms: float = 0.0
    channel: str = ""
    candidates: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "subject": self.claim.subject, "aspect": self.claim.aspect,
            "asserted": self.claim.value, "verdict": self.verdict,
            "detail": self.detail, "source_file": self.source_file,
            "source_location": self.source_location,
            "evidence": self.evidence, "latency_ms": round(self.latency_ms, 1),
            "channel": self.channel,
        }


class Verifier:
    """Adjudicates claims using retrieval only. Holds no file handles."""

    def __init__(self, retriever, embed_fn: Callable[[str], list[float]] | None = None,
                 domain: str = "pilot", top: int = 10, tolerance: float = 0.10,
                 aspect_coverage: float = 0.6):
        self.retriever = retriever
        self.embed_fn = embed_fn
        self.domain = domain
        self.top = top
        #: Relative tolerance when comparing figures. Cookbooks round, and
        #: "454 g" against "1 lb" must not read as a contradiction.
        self.tolerance = tolerance
        #: Fraction of an aspect's discriminating words a passage must contain
        #: before it is allowed to settle the claim.
        self.aspect_coverage = aspect_coverage

    @staticmethod
    def _names(subject: str, body: str) -> bool:
        """Is the passage actually about this preparation?

        A multi-word subject is checked as a phrase first. "white roux" spread
        across a page about mayonnaise is not that page being about white roux,
        and requiring only that both words appear somewhere accepted exactly
        that.
        """
        subj = norm(subject)
        if not subj:
            return True
        if subj in body:
            return True
        words = [w for w in subj.split() if len(w) > 3 and w not in _WEAK]
        if not words:
            return False
        if len(words) == 1:
            return words[0] in body
        # Words scattered across a page are not that page being about the
        # subject: "white" and "roux" both appear on a page about mayonnaise.
        # Require them near each other instead — which still accepts a table
        # row reading "Mornay  Gruyère and Parmesan", where the exact phrase
        # "mornay sauce" never occurs.
        anchor = max(words, key=len)
        rest = [w for w in words if w != anchor]
        for m in re.finditer(re.escape(anchor), body):
            window = body[max(0, m.start() - _NEAR): m.end() + _NEAR]
            if all(w in window for w in rest):
                return True
        return False

    # -- evidence ---------------------------------------------------------
    def passages(self, query: str, must: str = "") -> tuple[list[dict], str]:
        res = self.retriever.query(
            query, embed_fn=self.embed_fn, channels=("vector", "fulltext", "graph"),
            hops=1, domain=self.domain, must=must)
        if res.refused or not res.hits:
            return [], "refused" if res.refused else "empty"
        ids = [h.node_id for h in res.hits[: self.top]]
        props = self.retriever.hydrate(ids)
        ordered = [props[i] for i in ids if i in props]
        channel = "fast" if res.channel_counts.get("fast_path") else "hybrid"
        return ordered, channel

    @staticmethod
    def _window(text: str, claim: "Claim", width: int = 420) -> str:
        """The part of the passage that settles the claim, not its first lines.

        A page node carries its whole page; quoting from character zero puts a
        running header in front of the reader and leaves the deciding sentence
        out of view.
        """
        body = norm(text)
        anchors = [w for w in (norm(claim.aspect) + " " + norm(claim.value)).split()
                   if len(w) > 3 and w not in _WEAK]
        best = -1
        for w in anchors:
            i = body.find(w)
            if i >= 0 and (best < 0 or i < best):
                best = i
        if best < 0:
            return text[:width]
        lo = max(0, best - width // 3)
        return text[lo: lo + width]

    # -- adjudication -----------------------------------------------------
    def _match(self, claim: Claim, text: str) -> tuple[bool, str]:
        """Does `text` speak to this claim, and does it agree?

        Two gates, and both are needed. The subject must be named, or the
        passage is about a different preparation and any figure in it is
        unrelated — a Hollandaise recipe mentions egg yolks, which says nothing
        about whether a Mornay contains them. The aspect must then be mostly
        present, not merely brushed: requiring *any* word let "Grana Padano
        cheese" pass on the word "cheese" alone and every claim on the card came
        back confirmed, including the false ones.
        """
        body = norm(text)
        if not self._names(claim.subject, body):
            return False, ""
        terms = [w for w in norm(claim.aspect).split()
                 if len(w) > 3 and w not in _WEAK]
        # A claim with a figure is discriminated by that figure, so a partly
        # matching aspect is enough to locate it. A claim WITHOUT one has
        # nothing else to go on: "roux equal parts butter and flour" matched a
        # passage carrying "roux", "butter" and "flour" but neither "equal" nor
        # "parts" — three terms out of five, and the two it missed were the
        # entire assertion. Presence claims therefore require all of them.
        needed = self.aspect_coverage if claim.value else 1.0
        if terms:
            present = sum(1 for w in terms if w in body)
            if present / len(terms) < needed:
                return False, ""
        if not claim.value:
            return True, f"presente ({present}/{len(terms)} termini)" if terms else "presente"
        want = quantities(claim.value)
        if not want:
            return (norm(claim.value) in body), "text"
        found = quantities(text)
        if not found:
            return False, ""
        for wv, wu in want:
            for fv, fu in found:
                if fu != wu:
                    continue
                if abs(fv - wv) <= self.tolerance * max(wv, 1e-9):
                    return True, f"{fv:g} {fu} ≈ {wv:g} {wu}"
        # Same unit present, different magnitude: the source disagrees.
        same_unit = [f"{fv:g} {fu}" for fv, fu in found if fu == want[0][1]]
        return False, ("differs: source says " + ", ".join(same_unit[:4])
                       if same_unit else "")

    def check(self, claim: Claim) -> Finding:
        import time
        t0 = time.perf_counter()
        # The subject anchors retrieval: it is what the claim is about.
        docs, channel = self.passages(claim.query, must=claim.subject)
        if not docs:
            return Finding(claim, NOT_FOUND, detail="nothing retrieved",
                           latency_ms=(time.perf_counter() - t0) * 1000, channel=channel)
        disagreement = None
        for d in docs:
            text = d.get("passage") or d.get("text_excerpt") or ""
            ok, why = self._match(claim, text)
            if ok:
                return Finding(claim, SUPPORTED, evidence=self._window(text, claim),
                               source_file=d.get("source_file", ""),
                               source_location=d.get("source_location", ""),
                               detail=why,
                               latency_ms=(time.perf_counter() - t0) * 1000,
                               channel=channel)
            if why.startswith("differs") and disagreement is None:
                disagreement = (d, why)
        if disagreement:
            d, why = disagreement
            return Finding(claim, CONTRADICTED,
                           evidence=self._window(d.get("passage") or "", claim),
                           source_file=d.get("source_file", ""),
                           source_location=d.get("source_location", ""),
                           detail=why,
                           latency_ms=(time.perf_counter() - t0) * 1000, channel=channel)
        return Finding(claim, NOT_FOUND, detail="retrieved, but nothing addresses it",
                       source_file=docs[0].get("source_file", ""),
                       latency_ms=(time.perf_counter() - t0) * 1000, channel=channel)

    def check_all(self, claims: list[Claim]) -> list[Finding]:
        return [self.check(c) for c in claims]
