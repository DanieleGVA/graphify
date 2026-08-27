"""Settle a documentary claim against the graph — never against the source file.

The motivating case: a kitchen hands over a recipe card that says "reference
followed: <book>", and someone has to decide whether each figure on it actually
matches that book. Doing it by reading the PDF is what the system exists to
replace, so this module talks only to Neo4j. If a fact is not in the graph the
answer is "not found", never "let me open the book" — an unsupported claim is a
finding, not a reason to fall back.

Five verdicts (ADR-0004 Q3), each machine-detectable — a doctrine state in a
prose field cannot be asserted on:

  SUPPORTED    the source states this value for this subject
  CONTRADICTED the source states a DIFFERENT value for it — the useful case,
               and the one a similarity search alone will never surface,
               because the contradicting passage is the most similar one
  NOT_FOUND    the corpus does not speak to it. Reported plainly rather than
               resolved by picking the nearest passage.
  UNPARSED     the claim carries a figure the grammar cannot read. Explicit
               refusal — the substring fall-through that once confirmed
               "1 gal" off a yield line is gone.
  CONFLICTED   one document supports and another contradicts. Both are cited
               and returned together, never resolved by rank order: the old
               first-SUPPORTED-wins loop was the silent side-pick the
               doctrine forbids.

Claims that carry a parseable figure are settled against the canonical
quantity facts (graphify_ent.facts, ADR-0004 Q1-B): every figure with its
declared owner, derived once per node from the stored passage — not
re-guessed per claim from raw text. Presence claims keep the lexical path.

Every finding carries the passage it was decided from, with book and page, so a
human can overrule it in one glance. Nothing here paraphrases the source.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["Claim", "Finding", "Verifier", "SUPPORTED", "CONTRADICTED",
           "NOT_FOUND", "UNPARSED", "CONFLICTED"]

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_FOUND = "NOT_FOUND"
UNPARSED = "UNPARSED"
CONFLICTED = "CONFLICTED"

#: Quantities as cookbooks write them: "454 g", "8 oz", "4.80 L", "1 lb 2 oz"
#: — and temperatures, "165°F/74°C". Temperatures were absent until a card
#: asserting 74 °C for ground beef came back SUPPORTED: with no unit matched,
#: the check fell through to a TEXT search and found "74°C" on the same page,
#: on the line about ground poultry. A figure the system cannot parse is a
#: figure it cannot contradict.
_QTY = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(kg|g|mg|lb|oz|l|litres?|liters?|ml|cl|dl|tbsp|tsp"
    r"|°\s*[CF]|degrees?\s*[CF])\b", re.I)

#: Everything normalised to grams or millilitres so "8 oz" and "227 g" compare.
_TO_BASE = {
    "kg": ("g", 1000.0), "g": ("g", 1.0), "mg": ("g", 0.001),
    "lb": ("g", 453.592), "oz": ("g", 28.3495),
    "l": ("ml", 1000.0), "litre": ("ml", 1000.0), "litres": ("ml", 1000.0),
    "liter": ("ml", 1000.0), "liters": ("ml", 1000.0),
    "ml": ("ml", 1.0), "cl": ("ml", 10.0), "dl": ("ml", 100.0),
    "tbsp": ("ml", 15.0), "tsp": ("ml", 5.0),
    # Temperatures normalise to Celsius; Fahrenheit needs an offset, so it is
    # handled in `quantities` rather than by a factor.
    "°c": ("c", 1.0), "°f": ("c", None),
}

#: Absolute tolerance for temperatures, in Celsius. The relative tolerance used
#: for weights would call 71 °C and 74 °C the same figure — a 4% gap that is
#: precisely the difference between the ground-beef row and the poultry row.
TEMP_TOLERANCE_C = 2.0


def quantities(text: str) -> list[tuple[float, str]]:
    """Every quantity in `text`, normalised to grams, millilitres or Celsius."""
    out = []
    for value, unit in _QTY.findall(text or ""):
        u = re.sub(r"[\s°]", "", unit.lower()).replace("degrees", "").replace("degree", "")
        v = float(value.replace(",", "."))
        if u == "c":
            out.append((v, "c"))
            continue
        if u == "f":
            out.append(((v - 32) * 5 / 9, "c"))
            continue
        base = _TO_BASE.get(u)
        if not base or base[1] is None:
            continue
        out.append((v * base[1], base[0]))
    return out


#: Words that carry no discriminating power in a recipe corpus — requiring
#: them would reject good passages, counting them would inflate coverage.
_WEAK = {"sauce", "cheese", "with", "into", "from", "that", "this", "quantity",
         "quantita",
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
    #: The claim as it was actually adjudicated ("aspect→['milk']; value
    #: '1 gal'→3785 ml"). Zero silent rewrites: a reviewer can contest the
    #: rewrite, not only the verdict.
    normalized: str = ""
    #: CONFLICTED only: both readings, each with its own citation.
    conflict: dict | None = None

    def as_dict(self) -> dict:
        out = {
            "subject": self.claim.subject, "aspect": self.claim.aspect,
            "asserted": self.claim.value, "verdict": self.verdict,
            "detail": self.detail, "source_file": self.source_file,
            "source_location": self.source_location,
            "evidence": self.evidence, "latency_ms": round(self.latency_ms, 1),
            "channel": self.channel, "normalized": self.normalized,
        }
        if self.conflict is not None:
            out["conflict"] = self.conflict
        return out


class Verifier:
    """Adjudicates claims using retrieval only. Holds no file handles."""

    def __init__(self, retriever, embed_fn: Callable[[str], list[float]] | None = None,
                 domain: str = "pilot", top: int = 10, tolerance: float = 0.10,
                 aspect_coverage: float = 0.6,
                 glossary: dict[str, list[str]] | None = None):
        self.retriever = retriever
        self.embed_fn = embed_fn
        self.domain = domain
        self.top = top
        #: The 6.0 language-variant glossary, the SAME object HybridRetriever
        #: takes (ADR-0004 Q2) — verify carries no synonym map of its own.
        #: term -> variants, indexed both ways: "latte" reaches "milk".
        self._gloss: dict[str, set[str]] = {}
        for term, alts in (glossary or {}).items():
            fam = {norm(term), *(norm(a) for a in alts)}
            for w in fam:
                self._gloss.setdefault(w, set()).update(fam)
        #: Facts are derived once per node and cached — the ADR's interim for
        #: T60 materialization: same function, same results, no re-guessing.
        self._fact_cache: dict[str, list] = {}
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

    #: Half-width of the window read around the subject when a claim carries a
    #: figure. Wide enough for a sentence that names its subject once and its
    #: quantity a clause later.
    NEAR_WINDOW = 320

    #: How many lines above a table row are read as its headers, and how many
    #: lines a row spans. A PDF table puts every cell on its own line, so a row
    #: is name + figure + description, and its family header sits a few rows up.
    TABLE_CONTEXT = 6
    TABLE_ROW = 3

    @staticmethod
    def _is_tabular(text: str) -> bool:
        """A page that lists many figures against many names. There a number
        found "somewhere on the page" belongs to some other row.

        Counting figures alone is not enough: a food-safety page discusses half
        a dozen temperatures in ordinary sentences and was read as a table, so
        the row logic pulled the two lines after a sentence and found a figure
        from the next paragraph. What distinguishes a table in extracted PDF
        text is that each cell is its own line — a line whose whole content IS
        the figure ("160°F/71°C"). Prose never produces those.
        """
        if len(_QTY.findall(text or "")) < 6:
            return False
        cells = 0
        for ln in (text or "").splitlines():
            rest = re.sub(r"[\s/,;:·|–—-]", "", _QTY.sub("", ln))
            if ln.strip() and not rest:
                cells += 1
        return cells >= 3

    def _near_subject(self, subject: str, text: str) -> str:
        """The part of `text` that speaks about this subject.

        On a table the unit is the LINE: "Turkey, chicken 165°F/74°C" and
        "Beef, veal, lamb, pork 160°F/71°C" sit two lines apart, and any
        character window wide enough to hold a row's own figure also holds its
        neighbour's. Measured: reading by window confirmed 74 °C for ground
        beef from the poultry line. In prose, where a subject and its number
        can be a clause apart, the window is the right unit — so the shape of
        the text decides which is used.
        """
        subj = norm(subject)
        words = [w for w in subj.split() if len(w) > 3 and w not in _WEAK] or [subj]
        if not any(words):
            return text
        if self._is_tabular(text):
            # In a PDF table each cell is its own line: the row reads
            # "Beef, veal, lamb, pork" and the figure "160°F/71°C" sits on the
            # NEXT line. So a row is the matching line and the two that follow.
            #
            # No single word of the subject picks that row, because the table is
            # NESTED — a family header governs several rows:
            #
            #     Ground meat and meat mixtures      <- "ground" lives here
            #       Turkey, chicken       165°F/74°C
            #       Beef, veal, lamb, pork  160°F/71°C   <- "beef" lives here
            #
            # Anchoring on the longest word ("ground") reads the poultry row and
            # confirms 74 °C for ground beef; anchoring on the rarest picks the
            # header, which is the same row again — measured, both ways. What
            # identifies the row is the two words TOGETHER: score each line by
            # how much of the subject its own text plus the headers above it
            # account for, and keep the lines that account for the most. A line
            # that contributes nothing of the subject ("Turkey, chicken") is not
            # a candidate at all, and when several tie the caller still sees
            # several figures and declares the ambiguity.
            lines = [ln for ln in text.splitlines() if ln.strip()]
            best, hits = 0, []
            for i, ln in enumerate(lines):
                if not any(w in norm(ln) for w in words):
                    continue
                ctx = norm("\n".join(lines[max(0, i - self.TABLE_CONTEXT): i + 1]))
                cover = sum(1 for w in words if w in ctx)
                if cover > best:
                    best, hits = cover, [i]
                elif cover == best:
                    hits.append(i)
            picked: list[str] = []
            for i in hits:
                picked.extend(lines[i: i + self.TABLE_ROW])
            return "\n".join(picked)
        out = []
        body = norm(text)
        anchor = max(words, key=len)
        for m in re.finditer(re.escape(anchor), body):
            lo = max(0, m.start() - self.NEAR_WINDOW)
            out.append(text[lo: m.end() + self.NEAR_WINDOW])
        return "\n".join(out) if out else ""

    def _sentences_about(self, subject: str, text: str, also: str = "") -> str:
        """The sentences that name the subject, and nothing else.

        Tighter than the character window and often decisive: a food-safety
        page states its rule in one sentence — "reheated to at least 165°F/74°C
        for a minimum of 15 seconds" — while the paragraph after it discusses a
        different temperature. A window wide enough for prose reaches that
        paragraph and the two figures then look like a table's ambiguity.
        """
        words = [w for w in norm(subject).split() if len(w) > 3 and w not in _WEAK]
        if not words:
            return ""
        anchor = max(words, key=len)
        parts = re.split(r"(?<=[.!?])[\s ]+", text or "")
        # What the aspect adds BEYOND the subject. "reheated foods" / "reheated
        # to at least" share their first word, and keeping it would let every
        # sentence about reheating through — including "holds reheated foods
        # above 135°F/57°C", which is a different rule with a different figure.
        # The discriminating word is "least".
        extra = [w for w in norm(also).split()
                 if len(w) > 3 and w not in _WEAK and w not in words]
        keep = []
        for p in parts:
            low = norm(p)
            if anchor not in low:
                continue
            # The aspect is what says WHICH relation is meant. "reheated to at
            # least" and "brought to the proper temperature" are two sentences
            # of the same page, both about reheating, with two different
            # figures: without the aspect the scope holds both and the guard
            # below correctly calls it ambiguous — correctly, and uselessly.
            if extra and not any(w in low for w in extra):
                continue
            keep.append(p)
        return "\n".join(keep)

    def _scopes(self, claim: "Claim", text: str) -> list[str]:
        """Places to read a figure, tightest first.

        Adjudication takes the first one that offers a figure in the unit the
        claim uses. Tightest-first is the whole discipline: a wider scope is a
        weaker claim about which figure belongs to which subject, and widening
        past what actually names the subject is how a number from the next
        paragraph — or the next table row — gets read as an answer.
        """
        subject = claim.subject
        if self._is_tabular(text):
            row = self._near_subject(subject, text)
            # A table has no prose to fall back on: a figure outside the row is
            # another row's, never this one's.
            return [row] if row else [text]
        near = self._near_subject(subject, text)
        scopes = [self._sentences_about(subject, text, also=claim.aspect),
                  self._sentences_about(subject, text),
                  near]
        # Only when the subject cannot be located at all does the whole page
        # come back into play — `_names` is deliberately fuzzier than these
        # anchors, so that case exists and refusing it outright would lose
        # evidence the system used to find.
        return [s for s in scopes if s] or [text]

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
        # Read the figure NEAR the subject, not anywhere on the page. A page of
        # doneness temperatures names a dozen meats and a dozen figures; taking
        # any of them confirmed "ground beef at 74 °C" from the line about
        # ground poultry. The window is generous — prose separates a subject
        # from its number — but it is a window.
        # `_near_subject` was written for this line, with its measurements in
        # its docstring, and was never called: `quantities(text)` read the whole
        # page, so the ambiguity guard below declined every table — which is
        # exactly where a doneness figure lives. Two claims of the card set
        # failed that way with the right page already ranked first.
        unit = want[0][1]
        for scope in self._scopes(claim, text):
            found = [(fv, fu) for fv, fu in quantities(scope) if fu == unit]
            if not found:
                continue                       # nothing here; widen once
            # A doneness table lists many meats against many temperatures,
            # nested ("Ground meat and meat mixtures" over "Beef, veal, lamb,
            # pork"), and which row governs is structure the passage no longer
            # carries. When several DIFFERENT temperatures survive the tightest
            # scope that names the subject, saying "the source gives several
            # values" is the correct answer — picking one silently is the
            # failure this system exists to prevent. Measured: without it, a
            # card claiming 74 °C for ground beef was confirmed from the
            # poultry row. Restricted to temperatures, which is where it was
            # measured: an ingredient list also holds many figures, and there
            # the aspect names which one is meant.
            if unit == "c":
                temps = {round(v) for v, _ in found}
                if len(temps) > 1:
                    return False, ("ambiguous: the page gives several temperatures "
                                   f"({', '.join(str(t) for t in sorted(temps))} C)")
            for wv, wu in want:
                for fv, fu in found:
                    tol = (TEMP_TOLERANCE_C if wu == "c"
                           else self.tolerance * max(wv, 1e-9))
                    if wu == fu and abs(fv - wv) <= tol:
                        return True, f"{fv:g} {fu} ≈ {wv:g} {wu}"
            # Same unit present, different magnitude: the source disagrees.
            return False, ("differs: source says "
                           + ", ".join(f"{fv:g} {fu}" for fv, fu in found[:4]))
        return False, ""

    # -- canonical facts channel ------------------------------------------
    def _facts_for(self, doc: dict) -> list:
        from graphify_ent.facts import derive

        key = doc.get("id") or f"{doc.get('source_file')}|{doc.get('source_location')}"
        if key not in self._fact_cache:
            self._fact_cache[key] = derive(
                doc.get("passage") or doc.get("text_excerpt") or "")
        return self._fact_cache[key]

    def _variants(self, tok: str) -> set[str]:
        return self._gloss.get(tok, {tok})

    def _covers(self, tokens: list[str], anchor: str) -> float:
        body = norm(anchor)
        hit = sum(1 for t in tokens if any(v in body for v in self._variants(t)))
        return hit / len(tokens) if tokens else 0.0

    def _agrees(self, cqs, fact) -> bool:
        for q in cqs:
            if q.unit != fact.unit_base:
                continue
            tol = (TEMP_TOLERANCE_C if q.unit == "c"
                   else self.tolerance * max(q.hi, 1e-9))
            if fact.value_lo - tol <= q.hi and q.lo <= fact.value_hi + tol:
                return True
        return False

    def _address(self, claim: Claim, cqs, toks: list[str], facts: list,
                 passage: str = ""):
        """The facts of one document that speak to this claim, decided.

        The ASPECT names which figure is meant, so aspect-anchored facts
        decide first ("Milk" for a milk quantity — never the yield). Where no
        anchor names the aspect — the doneness table, whose aspect words are
        page-level — the SUBJECT keys the row instead, any token: "Beef,
        veal, lamb, pork" is how a table lists ground beef. An exact anchor
        (table-row) outranks a proximity one (ADR-0004 Q1).
        """
        units = {q.unit for q in cqs}
        cands = [f for f in facts if f.unit_base in units]
        if not cands:
            return None
        group = []
        if toks:
            # Coverage is read at two levels: the anchor must name at least
            # one discriminating token — the yield line names no ingredient,
            # which is what keeps "1 gal" out of a milk claim — and the REST
            # of the aspect may live anywhere in the passage. "Sugar for egg
            # whites" names two rows of one recipe: no single row carries
            # both words, and requiring it turned a true claim unverifiable
            # (measured, T74 re-baseline).
            page = norm(passage)
            hits = []
            for f in cands:
                if self._covers(toks, f.anchor_text) >= self.aspect_coverage:
                    hits.append(f)
                    continue
                anchored = [t for t in toks
                            if any(v in norm(f.anchor_text) for v in self._variants(t))]
                rest = [t for t in toks if t not in anchored]
                if anchored and all(
                        any(v in page for v in self._variants(t)) for t in rest):
                    hits.append(f)
            exact = [f for f in hits if f.anchor_kind == "table-row"]
            group = exact or hits
        if not group:
            subj = [w for w in norm(claim.subject).split()
                    if len(w) > 3 and w not in _WEAK]
            group = [f for f in cands
                     if any(any(v in norm(f.anchor_text) for v in self._variants(w))
                            for w in subj)]
            exact = [f for f in group if f.anchor_kind == "table-row"]
            group = exact or group
            if not group:
                return None
            # Measured doctrine: when a subject-level read finds SEVERAL
            # temperatures, which row governs is structure the claim cannot
            # recover — declare nothing rather than pick one.
            if "c" in units:
                temps = {round(f.value_lo) for f in group if f.unit_base == "c"}
                if len(temps) > 1:
                    return None
            agree = [f for f in group if self._agrees(cqs, f)]
            return (bool(agree), (agree or group)[0], group, "subject")
        agree = [f for f in group if self._agrees(cqs, f)]
        return (bool(agree), (agree or group)[0], group, "aspect")

    def _check_facts(self, claim: Claim, cqs, t0) -> Finding:
        import time

        toks = [w for w in norm(claim.aspect).split()
                if len(w) > 3 and w not in _WEAK]
        normalized = (f"aspect→{toks}; value {claim.value!r}→"
                      + ", ".join(f"{q.lo:g} {q.unit}" if q.single
                                  else f"{q.lo:g}–{q.hi:g} {q.unit}" for q in cqs))
        docs, channel = self.passages(claim.query, must=claim.subject)
        done = lambda: (time.perf_counter() - t0) * 1000
        if not docs:
            return Finding(claim, NOT_FOUND, detail="nothing retrieved",
                           normalized=normalized, latency_ms=done(), channel=channel)
        sup, con = [], []
        for d in docs:
            text = d.get("passage") or d.get("text_excerpt") or ""
            if not self._names(claim.subject, norm(text)):
                continue
            decided = self._address(claim, cqs, toks, self._facts_for(d), text)
            if decided is None:
                continue
            agreed, fact, group, how = decided
            (sup if agreed else con).append((d, fact, group, how))
        # A fact reached THROUGH THE ASPECT is a stronger read than one
        # reached through the subject fallback — the same outranking the ADR
        # states for anchors, one level up. Measured: the steam-table rule
        # (57°C, subject-matched off a narrow node) manufactured a conflict
        # with the reheat rule (74°C, aspect-matched) — two different rules,
        # not two readings of one.
        if any(how == "aspect" for *_, how in sup + con):
            sup = [x for x in sup if x[3] == "aspect"]
            con = [x for x in con if x[3] == "aspect"]

        def cite(d, f):
            return {"source_file": d.get("source_file", ""),
                    "source_location": d.get("source_location", ""),
                    "evidence": f.raw_text,
                    "figure": f"{f.value_lo:g} {f.unit_base}"}

        if sup and con:
            # Both readings exist. Returned together, flagged, never resolved
            # by rank order — the claim-level form of the CONTRADICTS
            # guarantee (blueprint §3 controls 3/4).
            a, b = cite(*sup[0][:2]), cite(*con[0][:2])
            return Finding(claim, CONFLICTED,
                           evidence=sup[0][1].raw_text,
                           source_file=a["source_file"],
                           source_location=a["source_location"],
                           detail=(f"for: {a['figure']} @ {a['source_location']}; "
                                   f"against: {b['figure']} @ {b['source_location']}"),
                           normalized=normalized,
                           conflict={"for": a, "against": b},
                           latency_ms=done(), channel=channel)
        if sup:
            d, f, _, _ = sup[0]
            return Finding(claim, SUPPORTED, evidence=f.raw_text,
                           source_file=d.get("source_file", ""),
                           source_location=d.get("source_location", ""),
                           detail=(f"{f.value_lo:g} {f.unit_base} ≈ claim "
                                   f"[{f.anchor_kind}]"),
                           normalized=normalized, latency_ms=done(), channel=channel)
        if con:
            d, f, group, _ = con[0]
            vals = ", ".join(f"{g.value_lo:g} {g.unit_base}" for g in group[:4])
            return Finding(claim, CONTRADICTED, evidence=f.raw_text,
                           source_file=d.get("source_file", ""),
                           source_location=d.get("source_location", ""),
                           detail=f"differs: source says {vals} [{f.anchor_kind}]",
                           normalized=normalized, latency_ms=done(), channel=channel)
        return Finding(claim, NOT_FOUND, detail="retrieved, but no fact addresses it",
                       source_file=docs[0].get("source_file", ""),
                       normalized=normalized, latency_ms=done(), channel=channel)

    def check(self, claim: Claim) -> Finding:
        import time

        from graphify_ent.quantities import default_table, scan

        t0 = time.perf_counter()
        if claim.value:
            cqs = scan(claim.value, default_table())
            if cqs:
                # A figure the grammar reads is settled against canonical
                # facts — figures with declared owners — never against raw
                # page text (ADR-0004 Q1/Q3).
                return self._check_facts(claim, cqs, t0)
            if re.search(r"\d", claim.value):
                # A figure the grammar cannot parse is a figure this module
                # cannot adjudicate — and must never fall through to the
                # substring search below, which confirmed "1 gal" as a milk
                # quantity off the yield line (ADR-0004 Q4-A).
                return Finding(claim, UNPARSED,
                               detail=f"unparsed figure: {claim.value!r} — refusing, not guessing",
                               latency_ms=(time.perf_counter() - t0) * 1000)
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
