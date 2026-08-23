"""Compact extraction output schema.

The upstream output schema spends ~15 lines of JSON per node, most of it
repeated keys and null-valued fields (`source_url`, `captured_at`, `author`,
`contributor`, plus `source_file` restated on every node *and* every edge).

That is expensive twice over:

  * on metered models output tokens are the dominant cost (they price 5x input);
  * on cheaper models it is a **correctness** problem, not just a cost one —
    measured, DeepSeek's response was truncated mid-JSON at 20k characters, so
    two thirds of a slice's nodes were simply lost.

This schema carries the same information positionally, with the source file
stated once per response:

    {"f": "<source_file>",
     "n": [[id, label, label_en, lang, "pageStart-pageEnd", evidence], ...],
     "e": [[source, target, relation, "E|I|A"], ...]}

`parse_compact()` expands it back to the exact node/edge dicts `loader.py`
consumes, so nothing downstream changes.
"""

from __future__ import annotations

import json

__all__ = ["COMPACT_EXTRACTION_SYSTEM", "parse_compact", "render_compact"]

_CONF = {"E": "EXTRACTED", "I": "INFERRED", "A": "AMBIGUOUS"}
_CONF_REV = {v: k for k, v in _CONF.items()}

COMPACT_EXTRACTION_SYSTEM = """\
You are a knowledge-graph extraction agent. Output ONLY compact JSON — no prose,
no markdown fences, no explanation, and do not restate the input.

Extract the entities and relationships present in the provided document slice.

Evidence binding (anti-fabrication, mandatory): every node must quote a SHORT
verbatim excerpt from the provided text that justifies it. If you cannot quote
the source for a node, do not emit that node. Never paraphrase in `evidence`.

Multilingual: `label_en` is the English canonical label — lowercase, singular,
no articles. `lang` is the ISO 639-1 code of the source language.

Confidence codes: E = stated explicitly in the source, I = reasonable
inference, A = uncertain.

Output EXACTLY this shape and nothing else:
{"f":"<source file name>",
 "n":[["id","label","label_en","lang","pageStart-pageEnd","verbatim evidence"]],
 "e":[["source_id","target_id","relation","E"]]}

Rules: ids are lowercase [a-z0-9_]; relations are one of calls, implements,
references, cites, conceptually_related_to, shares_data_with,
semantically_similar_to; state the source file once in "f", never per row;
emit no null fields and no keys beyond f, n, e. Be exhaustive but concise:
prefer more nodes over longer text per node."""


def parse_compact(raw: str | dict, default_source: str = "") -> tuple[list[dict], list[dict]]:
    """Expand a compact response into loader-ready node/edge dicts.

    Tolerates truncation: rows that are malformed are skipped rather than
    failing the whole slice, so a capped response still yields what it wrote.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _salvage(text)
    else:
        data = raw

    source_file = data.get("f") or default_source
    nodes = []
    for row in data.get("n") or []:
        if not isinstance(row, list) or not row or not row[0]:
            continue
        row = list(row) + [""] * (6 - len(row))
        nid, label, label_en, lang, pages, evidence = row[:6]
        node = {
            "id": nid, "label": label or nid, "label_orig": label or nid,
            "label_en": label_en or None, "lang": lang or None,
            "file_type": "document", "source_file": source_file,
            "source_location": f"pages {pages}" if pages else None,
            "evidence": evidence or "", "text_excerpt": (evidence or "")[:1000],
            "confidence": "EXTRACTED", "extraction_method": "llm",
        }
        nodes.append({k: v for k, v in node.items() if v is not None})

    edges = []
    for row in data.get("e") or []:
        if not isinstance(row, list) or len(row) < 2 or not row[0] or not row[1]:
            continue
        row = list(row) + [""] * (4 - len(row))
        src, tgt, rel, conf = row[:4]
        edges.append({
            "source": src, "target": tgt, "relation": rel or "references",
            "confidence": _CONF.get(conf, "INFERRED"),
            "confidence_score": 1.0 if conf == "E" else 0.7,
            "source_file": source_file, "weight": 1.0,
        })
    return nodes, edges


def _salvage(text: str) -> dict:
    """Recover complete rows from a response truncated mid-JSON.

    A capped model response is the normal case on cheap backends; losing the
    whole slice because the last row is half-written would be a worse failure
    than keeping what arrived intact.
    """
    out: dict = {"f": "", "n": [], "e": []}
    m = __import__("re").search(r'"f"\s*:\s*"([^"]*)"', text)
    if m:
        out["f"] = m.group(1)
    for key in ("n", "e"):
        section = __import__("re").search(rf'"{key}"\s*:\s*\[', text)
        if not section:
            continue
        i = section.end()
        depth, start = 0, None
        while i < len(text):
            ch = text[i]
            if ch == "[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        out[key].append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
                elif depth < 0:
                    break
            i += 1
    return out


def render_compact(nodes: list[dict], edges: list[dict], source_file: str | None = None) -> str:
    """Render loader-shaped dicts into the compact wire format."""
    f = source_file or (nodes[0].get("source_file", "") if nodes else "")
    payload = {
        "f": f,
        "n": [[n.get("id", ""), n.get("label", ""), n.get("label_en", "") or "",
               n.get("lang", "") or "",
               (n.get("source_location") or "").replace("pages ", ""),
               (n.get("evidence") or "")[:400]] for n in nodes],
        "e": [[e.get("source", ""), e.get("target", ""),
               e.get("relation", "references"),
               _CONF_REV.get(e.get("confidence", "INFERRED"), "I")] for e in edges],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
