"""Phase A filter parser — deterministic, no model loaded.

The design 14 spec describes a Qwen3-4B-with-constrained-JSON parser as
the eventual filter parser. For the off-the-shelf architecture spike
(slice 14.0) we avoid pulling in a 4B-parameter LLM just to extract
structured constraints from short queries — a regex/keyword pass over
the existing per-domain attribute schema gets us 80% of the wins for
~1% of the latency.

The parser emits a list of dicts shaped:

    [{"attribute": <str>, "op": <str>, "value": <str|int|float>}, ...]

which is exactly the shape `arena_core.build_prefilter_sql` expects.

Phase B (LLM filter parser) lives behind an `enable_llm_parser` config
flag — easy to add without changing the pipeline shape.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Per-domain rule tables.
#
# These mirror the attribute_defs table in the existing SQLite schema
# (see implementations/design_05_sql/schema.py — DOMAIN_ATTRIBUTE_DEFS).
# Keep in sync; the eval gate will catch drift.
# ---------------------------------------------------------------------------
_SKI_TERRAIN_KEYWORDS = {
    "powder": "powder",
    "freeride": "freeride",
    "freestyle": "freestyle",
    "park": "park",
    "all-mountain": "all-mountain",
    "all mountain": "all-mountain",
    "frontside": "carving",
    "carving": "carving",
    "race": "race",
    "racing": "race",
    "on-piste": "on-piste",
    "on piste": "on-piste",
    "off-piste": "off-piste",
    "off piste": "off-piste",
    "groomed": "groomed",
    "touring": "touring",
    "backcountry": "backcountry",
}

_SKI_STIFFNESS_RULES = [
    # (pattern, op, attribute, value_on_match)
    # Negative-form rules use a lookbehind to exclude the positive match.
    # The "not playful" rule must short-circuit "playful".
    (r"(?<!not )\bvery stiff\b", "gte", "stiffness", 8),
    (r"(?<!not )\bstiff\b", "gte", "stiffness", 7),
    (r"\bsoft(?:er)?\b", "lte", "stiffness", 4),
    (r"(?<!not )\bdamp\b", "gte", "damp", 7),
    (r"\bnot playful\b", "lte", "playfulness", 4),
    (r"(?<!not )\bplayful\b", "gte", "playfulness", 7),
    (r"\bunforgiving\b", "lte", "forgiveness", 4),
    (r"(?<!un)\bforgiving\b", "gte", "forgiveness", 7),
    (r"\bedge grip\b", "gte", "edge_grip", 7),
    (r"\bstable\b", "gte", "stability_at_speed", 7),
    (r"\bgood (?:powder )?float\b", "gte", "powder_float", 7),
    (r"\bbeginner\b", "lte", "stiffness", 4),
    (r"\bbeginner\b", "gte", "forgiveness", 7),
]

_SKI_NUMERIC_RULES = [
    # (pattern, op, attribute) — value comes from the regex group.
    (r"(?:over|above|>=?|at least)\s*(\d{2,3})\s*mm(?:\s*(?:waist|underfoot))?",
     "gte", "waist_width_mm"),
    (r"(?:under|below|<=?|less than)\s*(\d{2,3})\s*mm(?:\s*(?:waist|underfoot))?",
     "lte", "waist_width_mm"),
    (r"(\d{2,3})\s*mm\+",                   "gte", "waist_width_mm"),
    (r"\baround\s*(\d{2,3})\s*-?\s*(\d{2,3})?\s*mm\b",  "range", "waist_width_mm"),
    # Two-number ranges, e.g. "80-95mm", "around 96-100mm waist"
    (r"\b(\d{2,3})\s*-\s*(\d{2,3})\s*mm\b", "range", "waist_width_mm"),
    # Bare "Nmm" alone — only emit when no range form already matched the
    # query (handled in _parse_ski). Default op is `eq` for backwards
    # compat but it's brittle when waists are sparse — see _parse_ski.
    (r"\b(\d{2,3})\s*mm\b",                 "eq",  "waist_width_mm"),
]

_SHOE_SURFACE_KEYWORDS = {
    "trail": "trail",
    "road": "road",
    "track": "track",
    "treadmill": "treadmill",
}

_SHOE_RULES = [
    (r"\bcushioned?\b",       "gte", "cushioning",     7),
    (r"\bresponsive\b",       "gte", "responsiveness", 7),
    (r"\bstable\b",           "gte", "stability",      7),
    (r"\bbreathable\b",       "gte", "breathability",  7),
    (r"\bdurable\b",          "gte", "durability",     7),
    (r"\blight(?:weight)?\b", "gte", "weight_feel",    7),
    (r"\bgrip\b",             "gte", "grip",           7),
]


def parse_query(query_text: str, domain: str) -> list[dict]:
    """Extract structured filter constraints from a free-text query."""
    text = (query_text or "").lower()
    if not text.strip():
        return []

    if domain == "ski":
        return _parse_ski(text)
    if domain == "running_shoe":
        return _parse_shoe(text)
    return []


_BALANCED_INTENT_RE = re.compile(
    r"\b(?:both\s+\w+\s+and\s+\w+|versatile|not\s+(?:a\s+)?specialist|"
    r"all[- ]rounder|do[- ]it[- ]all|one[- ]ski[- ]quiver|equally well|mixed conditions)\b"
)


def _parse_ski(text: str) -> list[dict]:
    filters: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()  # (attr, op) — at most one per pair

    # "Balanced" / "versatile" / "both X and Y" queries are explicitly
    # asking for non-specialist skis. Suppress narrowing terrain filters
    # — those would exclude all-mountain skis whose terrain list doesn't
    # include the literal keyword the user mentioned (e.g. SKI-006's
    # terrain is ["all-mountain", "on-piste", "off-piste"] with no
    # "powder", but it's the perfect answer for "handles both powder and
    # hardpack equally well").
    balanced_intent = bool(_BALANCED_INTENT_RE.search(text))

    # Terrain — emit a contains filter against the categorical column.
    if not balanced_intent:
        matched_terrains: list[str] = []
        for kw, canonical in _SKI_TERRAIN_KEYWORDS.items():
            if kw in text and canonical not in matched_terrains:
                matched_terrains.append(canonical)
        for terrain in matched_terrains:
            key = ("terrain", "contains")
            if key not in seen_keys:
                filters.append({"attribute": "terrain", "op": "contains", "value": terrain})
                seen_keys.add(key)

    # Numeric — pick the FIRST match for each (attr, op) pair so duplicate
    # mentions don't produce contradictory filters.
    range_attrs: set[str] = set()
    for pattern, op, attr in _SKI_NUMERIC_RULES:
        m = re.search(pattern, text)
        if not m:
            continue
        if op == "range":
            try:
                lo = int(m.group(1))
                hi_raw = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                hi = int(hi_raw) if hi_raw else lo + 5  # "around N" → ±5
            except (IndexError, ValueError):
                continue
            if hi < lo:
                lo, hi = hi, lo
            for sub_op, val in (("gte", lo), ("lte", hi)):
                key = (attr, sub_op)
                if key not in seen_keys:
                    filters.append({"attribute": attr, "op": sub_op, "value": val})
                    seen_keys.add(key)
            range_attrs.add(attr)
            continue
        # Skip a strict-equality filter for an attribute where ANY other
        # numeric filter already fired — eq=90 contradicts lte=90 and
        # narrows the prefilter unnecessarily.
        if op == "eq" and any(
            seen_attr == attr and seen_op != "eq"
            for seen_attr, seen_op in seen_keys
        ):
            continue
        key = (attr, op)
        if key not in seen_keys:
            try:
                val = int(m.group(1))
            except (IndexError, ValueError):
                continue
            filters.append({"attribute": attr, "op": op, "value": val})
            seen_keys.add(key)

    # Stiffness / damp / playfulness / etc. — keyword to scale-attribute rules.
    for pattern, op, attr, val in _SKI_STIFFNESS_RULES:
        if re.search(pattern, text):
            key = (attr, op)
            if key not in seen_keys:
                filters.append({"attribute": attr, "op": op, "value": val})
                seen_keys.add(key)

    return filters


def _parse_shoe(text: str) -> list[dict]:
    filters: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()

    for kw, canonical in _SHOE_SURFACE_KEYWORDS.items():
        if kw in text:
            key = ("surface", "contains")
            if key not in seen_keys:
                filters.append({"attribute": "surface", "op": "contains", "value": canonical})
                seen_keys.add(key)
                break  # one surface per query is plenty

    for pattern, op, attr, val in _SHOE_RULES:
        if re.search(pattern, text):
            key = (attr, op)
            if key not in seen_keys:
                filters.append({"attribute": attr, "op": op, "value": val})
                seen_keys.add(key)

    return filters


# ---------------------------------------------------------------------------
# FTS5 query tokenization.
# Separate concern from filter parsing but lives here because both drop
# stop-words and lowercase the input.
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "and", "or", "the", "with", "for", "of", "to", "in", "on",
    "is", "are", "was", "were", "be", "i", "want", "need", "looking", "ski",
    "skis", "shoe", "shoes",
}
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")


def tokenize_for_fts5(query_text: str) -> str:
    """Turn a free-text query into a safe FTS5 MATCH expression.

    FTS5 has its own query syntax (quotes, NEAR, AND/OR/NOT, etc). To
    avoid surprises we extract tokens, drop stop-words, quote each
    surviving token, and OR-join them. Caller passes the result
    straight to `arena_core.fts5_search`.

    Returns an empty string when the query has no usable tokens; caller
    should treat that as "skip the lexical track".
    """
    if not query_text:
        return ""
    tokens = [t.lower() for t in _TOKEN_RE.findall(query_text)]
    keep = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    if not keep:
        return ""
    # Quote each token so e.g. NOT/AND/OR can't bleed into FTS5 syntax.
    quoted = [f'"{t}"' for t in keep]
    return " OR ".join(quoted)
