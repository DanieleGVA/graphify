"""R4 (with R2 folded in) — matching a recipe by what it is.

Three criteria in the order the T96 experiment validated them, weights
untouched from the experiment (0.60 / 0.25 / 0.15 worked untuned; R5's split
calibrates them properly, never on the data that measures them):

  1. **ingredient fingerprint** — canonical ingredients and their proportions,
     weighted by RARITY. Identity lives in rarity, not in mass: the first
     scoring weighted by share of the recipe, and a wrong page beat the true
     reference 67-50 with 62 of its 67 points coming from water+sugar+cream
     (present on 843/1,088/683 of 1,578 pages), while marsala (18 pages),
     mascarpone (20) and coffee liqueur (12) — the actual signature —
     contributed crumbs. Water is 28% of the cake and says nothing; marsala is
     3% and says almost everything. With the IDF the ingredient-only rank went
     49 -> 3, the combined rank to 1, and a fantasy title still ranked 1.
  2. **procedure** — canonical techniques (T98 registry), set and sequence.
  3. **title** — a tiebreak, never the identity.

**No corpus segmentation** (the revision of 2026-08-27): retrieval already
finds the right pages — the evidence lane grounds 95.5% of the canon records —
so candidates come from the retriever and only THOSE pages are parsed, on the
fly. Segmenting 1,578 pages to compare 40 was the work upside down.

The IDF is a property of the corpus. It is computed once per (domain, registry
version) over the corpus's recipe pages and cached as JSON next to the
evidence; the :Recipe graph layer that would hold it is deferred until R5 has
proven the matching (same revision).

`explain()` is not optional: a score nobody can read is a verdict nobody can
challenge, and the CONFORMS/DEVIATES report format exists precisely to be
checked against the page it cites.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from graphify_ent.recipes.ingredients import (
    Registry,
    Resolved,
    norm,
    parse_block,
    proportions,
)
from graphify_ent.recipes.techniques import similarity, techniques_in

__all__ = [
    "WEIGHTS",
    "Candidate",
    "CorpusIndex",
    "Match",
    "RecipeQuery",
    "explain",
    "match_text",
    "score_ingredients",
    "score_title",
]

#: T96's weights, untuned. R5 calibrates on half the books and reports on the
#: other half — never on the 200 records that measure the result.
WEIGHTS = (0.60, 0.25, 0.15)

#: A page is a candidate recipe when it resolves at least this many distinct
#: QUANTIFIED canonical ingredients — the bar T96 validated. Counting mere
#: presence let every page with "salt, oil and an onion" in prose into the
#: pool: 11,967 candidates against T96's 1,578, and the IDF diluted with them.
#: A recipe states amounts; that is what makes it one.
MIN_INGREDIENTS = 3


@dataclass
class RecipeQuery:
    """What we are looking for: a card or any recipe-shaped text, parsed."""

    title: str
    resolved: list[Resolved]
    proportions: dict[str, float]
    verbs: set = field(default_factory=set)
    verb_seq: list = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str, title: str = "",
                  registry: Registry | None = None) -> "RecipeQuery":
        resolved = parse_block(text, registry)
        method = text.split("Method:", 1)[1] if "Method:" in text else text
        return cls(title=title, resolved=resolved,
                   proportions=proportions(resolved),
                   verbs=techniques_in(method),
                   verb_seq=techniques_in(method, ordered=True))


@dataclass
class Candidate:
    """One corpus page, parsed the same way the query was."""

    source_file: str
    page: int
    proportions: dict[str, float]
    verbs: set
    verb_seq: list
    text: str = ""


@dataclass
class Match:
    candidate: Candidate
    combined: float
    s_ingredients: float
    s_procedure: float
    s_title: float

    def as_dict(self) -> dict:
        return {"book": self.candidate.source_file, "page": self.candidate.page,
                "combined": round(self.combined, 4),
                "ingredients": round(self.s_ingredients, 4),
                "procedure": round(self.s_procedure, 4),
                "title": round(self.s_title, 4)}


def score_ingredients(query: dict[str, float], cand: dict[str, float],
                      idf: dict[str, float]) -> float:
    """Rarity-weighted proportion agreement, in [0,1], scale-invariant.

    Normalised by the QUERY's own weighted mass: the score asks "how much of
    what identifies this recipe does the page account for", so a page that
    contains the recipe plus twenty others is not punished for its neighbours.
    """
    if not query:
        return 0.0
    num = sum(idf.get(k, 0.0) * min(query.get(k, 0.0), cand.get(k, 0.0))
              for k in set(query) | set(cand))
    den = sum(idf.get(k, 0.0) * v for k, v in query.items())
    return num / den if den else 0.0


def score_title(title: str, text: str) -> float:
    toks = re.findall(r"[a-zà-ÿ]{4,}", norm(title))
    if not toks:
        return 0.0
    body = norm(text)
    return sum(1 for t in toks if t in body) / len(toks)


class CorpusIndex:
    """The corpus's recipe pages, parsed once, with their IDF.

    Built from the graph's page nodes (the deterministic layer — no model ever
    wrote them) and cached as JSON keyed by (domain, registry version, page
    count): the IDF is a property of the corpus and recomputing 12,880 parses
    per query would be absurd, but an IDF that silently survives a corpus or
    vocabulary change is a stale denominator nobody notices — so the key
    changes with either.
    """

    def __init__(self, candidates: list[Candidate], idf: dict[str, float]):
        self.candidates = candidates
        self.idf = idf
        self._by_page = {(c.source_file, c.page): c for c in candidates}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_pages(cls, pages: list[tuple[str, int, str]],
                   registry: Registry | None = None) -> "CorpusIndex":
        cands: list[Candidate] = []
        for source_file, page, text in pages:
            resolved = parse_block(text, registry)
            props = proportions(resolved)
            named = {r.canonical for r in resolved}
            if len(props) < MIN_INGREDIENTS:
                continue
            # Presence without a figure still counts for the SET — under the
            # IDF, presence is half of identity — with a nominal share so the
            # rarity weight can see it without distorting the proportions.
            for canonical in named - set(props):
                props.setdefault(canonical, 0.0)
            cands.append(Candidate(source_file, page, props,
                                   techniques_in(text),
                                   techniques_in(text, ordered=True),
                                   text))
        return cls(cands, cls._idf(cands))

    @staticmethod
    def _idf(cands: list[Candidate]) -> dict[str, float]:
        df: dict[str, int] = {}
        for c in cands:
            for k in c.proportions:
                df[k] = df.get(k, 0) + 1
        n = max(len(cands), 1)
        return {k: math.log(n / max(v, 1)) for k, v in df.items()}

    @classmethod
    def from_graph(cls, loader, domain: str,
                   registry: Registry | None = None,
                   cache: Path | None = None) -> "CorpusIndex":
        reg = registry or Registry.load()
        with loader._session() as s:
            rows = [(r["f"], int(r["pg"]), r["p"] or "") for r in s.run(
                "MATCH (n:Entity) WHERE n.domain = $d "
                "AND n.extraction_method = 'page' "
                "RETURN n.source_file AS f, n.page_lo AS pg, n.passage AS p",
                d=domain)]
        key = f"{domain}:reg{reg.version}:{len(rows)}:v2"
        if cache and cache.exists():
            data = json.loads(cache.read_text())
            if data.get("key") == key:
                cands = [Candidate(c["f"], c["pg"], c["prop"],
                                   set(c["verbs"]), c["seq"], c.get("text", ""))
                         for c in data["candidates"]]
                return cls(cands, data["idf"])
        index = cls.from_pages(rows, reg)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # `text` travels with the candidate: leaving it out silently turned
            # the title criterion off — score_title fell back to the proportion
            # KEYS, so "Swiss Meringue" was searched inside "egg_white sugar".
            # For basic preparations the fingerprint is inherently ambiguous
            # (dozens of pages ARE egg whites and sugar) and the title exists
            # precisely to break those ties.
            cache.write_text(json.dumps(
                {"key": key,
                 "candidates": [{"f": c.source_file, "pg": c.page,
                                 "prop": c.proportions,
                                 "verbs": sorted(c.verbs), "seq": c.verb_seq,
                                 "text": c.text}
                                for c in index.candidates],
                 "idf": index.idf}, ensure_ascii=False))
        return index

    # -- matching ----------------------------------------------------------
    def rank(self, query: RecipeQuery,
             candidates: list[Candidate] | None = None,
             weights: tuple[float, float, float] = WEIGHTS) -> list[Match]:
        wi, wp, wt = weights
        out = []
        for c in (candidates if candidates is not None else self.candidates):
            si = score_ingredients(query.proportions, c.proportions, self.idf)
            sp = similarity(query.verbs, query.verb_seq, c.verbs, c.verb_seq)
            st = score_title(query.title, c.text)
            out.append(Match(c, wi * si + wp * sp + wt * st, si, sp, st))
        out.sort(key=lambda m: -m.combined)
        return out

    def page(self, source_file: str, page: int) -> Candidate | None:
        return self._by_page.get((source_file, page))


def match_text(text: str, index: CorpusIndex, title: str = "",
               registry: Registry | None = None,
               top: int = 10) -> list[Match]:
    """Card text in, ranked matches out — the whole R2+R4 surface."""
    return index.rank(RecipeQuery.from_text(text, title, registry))[:top]


def explain(query: RecipeQuery, match: Match, idf: dict[str, float]) -> str:
    """The score, readable — shared ingredients with their proportion deltas,
    what is missing, which techniques agree. The CONFORMS/DEVIATES report
    format, generated instead of verified."""
    c = match.candidate
    lines = [f"{c.source_file} p.{c.page}  combinato {match.combined:.3f} "
             f"(ingredienti {match.s_ingredients:.2f}, procedura "
             f"{match.s_procedure:.2f}, titolo {match.s_title:.2f})"]
    shared = sorted(set(query.proportions) & set(c.proportions),
                    key=lambda k: -idf.get(k, 0.0))
    for k in shared:
        qv, cv = query.proportions[k], c.proportions[k]
        verdict = "CONFORMS" if abs(qv - cv) <= 0.05 else \
            f"DEVIATES ({qv * 100:.1f}% vs {cv * 100:.1f}%)"
        lines.append(f"  = {k:<16} {verdict}  [rarita' {idf.get(k, 0.0):.2f}]")
    for k in sorted(set(query.proportions) - set(c.proportions),
                    key=lambda k: -idf.get(k, 0.0)):
        lines.append(f"  - {k:<16} assente dalla pagina "
                     f"[rarita' {idf.get(k, 0.0):.2f}]")
    both = sorted(query.verbs & c.verbs)
    if both:
        lines.append(f"  tecniche condivise: {', '.join(both)}")
    return "\n".join(lines)
