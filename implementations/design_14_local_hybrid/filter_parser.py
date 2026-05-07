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

Three sources of filters run for every domain:
  1. Domain-specific handler (`_parse_ski`, `_parse_shoe`) — hand-curated.
  2. Generic catalog-derived hooks (`_parse_generic`) — auto-discovered
     numeric ranges (price, points, etc.), categorical-value mentions
     (country, variety, region, ...), quality phrases (cheap, premium),
     and negation. No LLM.
  3. LLM-discovered phrase mappings loaded from `bundle/filter_phrases.json`
     (Layer 2). Optional; absent for legacy bundles.

Phase B (LLM filter parser) lives behind an `enable_llm_parser` config
flag — easy to add without changing the pipeline shape.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

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
    "twin tip": "park",        # twin tip is a park/freestyle feature
    "twin-tip": "park",
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
    "groomer": "groomed",
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
    (r"(\d{2,3})\s*mm(?:\s*waist)?\s+or\s+(?:wider|more|greater|larger|above)",
     "gte", "waist_width_mm"),
    (r"(\d{2,3})\s*mm(?:\s*waist)?\s+or\s+(?:narrower|less|smaller|under|below)",
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

# Construction terms — matched against the product_attributes
# `construction` text column (values like "titanal_sandwich",
# "carbon_paulownia", "graphene_sandwich", "composite_cap"). LIKE
# matching means partial substrings work, so "titanal" hits
# "titanal_sandwich".
_SKI_CONSTRUCTION_KEYWORDS = {
    "titanal":         "titanal",
    "metal":           "titanal",   # "metal in it" common synonym for titanal
    "carbon":          "carbon",
    "graphene":        "graphene",
    "composite cap":   "composite_cap",
    "composite":       "composite",
    "cap":             "cap",
    "race sandwich":   "race_sandwich",
    "maple":           "maple",
    "poplar":          "poplar",
    "paulownia":       "paulownia",
}

# Rocker-profile terms. Values map to the canonical
# rocker_profile column (camber | rocker_camber | full_rocker | etc.).
_SKI_ROCKER_KEYWORDS = {
    "full rocker":     "full_rocker",
    "full camber":     "camber",
    "pure camber":     "camber",
    "no rocker":       "camber",      # "no rocker" → require camber
}

# Brand match — the products table has a `brand` column already.
# Stored lowercase via parser canonicalization, matched against the
# product_attributes brand text column at retrieval time. We just emit
# a filter; the SQL contains-op joins via LIKE.
_SKI_BRAND_KEYWORDS = {
    "atomic":      "Atomic",
    "rossignol":   "Rossignol",
    "head":        "Head",
    "volkl":       "Volkl",
    "nordica":     "Nordica",
    "blizzard":    "Blizzard",
    "salomon":     "Salomon",
    "k2":          "K2",
    "dynastar":    "Dynastar",
    "black crows": "Black Crows",
    "dps":         "DPS",
    "moment":      "Moment",
    "elan":        "Elan",
    "fischer":     "Fischer",
    "armada":      "Armada",
    "line":        "Line",
    "faction":     "Faction",
}

# Phrases that imply the user wants to *exclude* an attribute. Used to
# detect "ski that does NOT use titanal", "without metal", "no twin tip".
# We look for these phrases preceding a construction / rocker / terrain
# / brand keyword within ~3 tokens.
_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:not|no|without|excludes?|except|never|isn'?t|doesn'?t(?:\s+(?:use|have))?)\b"
)
_NEGATED_RANGE = 35  # how many chars after a negation cue to scan

# Lifestyle / colloquial phrases that don't mention attributes directly
# but imply a strong attribute bias. The listwise reranker tends to fail
# on these because the surface form doesn't connect to the schema; a
# deterministic phrase->terrain map gives the prefilter and FTS5
# tokenizer enough signal to pull the right candidates into the top-20.
#
# Shape: phrase -> list of (terrain | rocker | construction) hints to
# emit as positive filters.
_SKI_LIFESTYLE_PHRASES: list[tuple[str, list[tuple[str, str]]]] = [
    # daily driver / one-quiver / do-it-all
    ("daily driver",       [("terrain", "all-mountain")]),
    ("do it all",          [("terrain", "all-mountain")]),
    ("do-it-all",          [("terrain", "all-mountain")]),
    ("quiver of one",      [("terrain", "all-mountain")]),
    ("one ski quiver",     [("terrain", "all-mountain")]),
    ("one-ski quiver",     [("terrain", "all-mountain")]),
    ("resort skier",       [("terrain", "all-mountain")]),
    ("resort skiing",      [("terrain", "all-mountain")]),
    # weather / conditions slang
    ("japow",              [("terrain", "powder")]),
    ("japan pow",          [("terrain", "powder")]),
    ("blower pow",         [("terrain", "powder")]),
    ("steep and deep",     [("terrain", "powder"), ("terrain", "big-mountain")]),
    ("ice coast",          [("terrain", "on-piste")]),
    ("east coast",         [("terrain", "on-piste")]),
    ("pnw",                [("terrain", "freeride")]),
    ("pacific northwest",  [("terrain", "freeride")]),
    ("crud",               [("terrain", "all-mountain")]),
    ("chunder",            [("terrain", "all-mountain")]),
    # use-style
    ("trees",              [("terrain", "all-mountain")]),
    ("tight trees",        [("terrain", "all-mountain")]),
    ("groomers at lunch",  [("terrain", "all-mountain")]),
    ("powder lap",         [("terrain", "off-piste")]),
    ("hits some powder",   [("terrain", "all-mountain")]),
    ("charging hard",      [("terrain", "freeride")]),
    ("charge hard",        [("terrain", "freeride")]),
    # tip/edge feel
    ("locks in",           [("terrain", "carving")]),
    ("locked in",          [("terrain", "carving")]),
    ("arcs cleanly",       [("terrain", "carving")]),
    ("clean arcs",         [("terrain", "carving")]),
    ("pop out",            [("terrain", "carving")]),
    ("pops out",           [("terrain", "carving")]),
    # progression / level
    ("grow into",          [("terrain", "all-mountain")]),
    ("progression",        [("terrain", "all-mountain")]),
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
    """Extract structured filter constraints from a free-text query.

    Returns a flat list of filter dicts. Negated filters carry a
    ``"_negate": True`` marker — callers (currently the design-14
    recommender) read this and apply those filters as a post-retrieval
    AND-exclusion rather than feeding them into the OR-based Rust
    prefilter (which can't narrow).
    """
    pos, neg = parse_query_with_negation(query_text, domain)
    return pos + [dict(f, _negate=True) for f in neg]


def parse_query_with_negation(
    query_text: str, domain: str
) -> tuple[list[dict], list[dict]]:
    """Split into (positive, negative) filter lists explicitly."""
    text = (query_text or "").lower()
    if not text.strip():
        return [], []

    if domain == "ski":
        pos, neg = _parse_ski(text)
    elif domain == "running_shoe":
        pos, neg = _parse_shoe(text), []
    else:
        pos, neg = [], []

    schema = _load_domain_schema(domain)
    if schema is not None:
        gen_pos, gen_neg = _parse_generic(text, schema)
        pos = _merge_filters(pos, gen_pos)
        neg = _merge_filters(neg, gen_neg)
        ph_pos, ph_neg = _apply_discovered_phrases(text, schema.discovered_phrases)
        pos = _merge_filters(pos, ph_pos)
        neg = _merge_filters(neg, ph_neg)

    return pos, neg


_NEGATION_CLAUSE_BREAKS = re.compile(r"[,;:]| and | but | while | yet | however ")


def _is_negated(text: str, span_start: int) -> bool:
    """Is the keyword at *span_start* preceded by a negation cue within
    ``_NEGATED_RANGE`` characters AND in the same clause (no comma /
    coordinating conjunction between the cue and the keyword)?"""
    window_start = max(0, span_start - _NEGATED_RANGE)
    window = text[window_start:span_start]
    if not _NEGATION_PREFIX_RE.search(window):
        return False
    # Clip the window forward to the last clause break — anything before
    # the break is in a different clause and shouldn't negate this one.
    last_break_end = 0
    for m in _NEGATION_CLAUSE_BREAKS.finditer(window):
        last_break_end = m.end()
    if last_break_end > 0:
        window = window[last_break_end:]
    return bool(_NEGATION_PREFIX_RE.search(window))


_BALANCED_INTENT_RE = re.compile(
    r"\b(?:both\s+\w+\s+and\s+\w+|versatile|not\s+(?:a\s+)?specialist|"
    r"all[- ]rounder|do[- ]it[- ]all|one[- ]ski[- ]quiver|equally well|mixed conditions)\b"
)


def _parse_ski(text: str) -> tuple[list[dict], list[dict]]:
    filters: list[dict] = []
    negative_filters: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()  # (attr, op, value) — dedupe

    def _add(target: list[dict], attr: str, op: str, value):
        key = (attr, op, str(value))
        if key in seen_keys:
            return
        seen_keys.add(key)
        target.append({"attribute": attr, "op": op, "value": value})

    # "Balanced" / "versatile" / "both X and Y" queries are explicitly
    # asking for non-specialist skis. Suppress narrowing terrain filters
    # — those would exclude all-mountain skis whose terrain list doesn't
    # include the literal keyword the user mentioned (e.g. SKI-006's
    # terrain is ["all-mountain", "on-piste", "off-piste"] with no
    # "powder", but it's the perfect answer for "handles both powder and
    # hardpack equally well").
    balanced_intent = bool(_BALANCED_INTENT_RE.search(text))

    # Lifestyle phrases — emit attribute hints inferred from colloquial
    # use-cases ("daily driver" → all-mountain, "japow" → powder).
    # Run before terrain matching so the listwise reranker sees the
    # right candidate set; phrases are positive-only (no negation).
    for phrase, hints in _SKI_LIFESTYLE_PHRASES:
        if phrase not in text:
            continue
        for attr, value in hints:
            _add(filters, attr, "contains", value)

    # Terrain — emit a contains filter against the categorical column.
    if not balanced_intent:
        matched_terrains: list[tuple[str, int]] = []
        for kw, canonical in _SKI_TERRAIN_KEYWORDS.items():
            idx = text.find(kw)
            if idx >= 0 and canonical not in {t for t, _ in matched_terrains}:
                matched_terrains.append((canonical, idx))
        for terrain, idx in matched_terrains:
            target = negative_filters if _is_negated(text, idx) else filters
            _add(target, "terrain", "contains", terrain)

    # Construction — match against the construction text column.
    for kw, canonical in _SKI_CONSTRUCTION_KEYWORDS.items():
        idx = text.find(kw)
        if idx < 0:
            continue
        target = negative_filters if _is_negated(text, idx) else filters
        _add(target, "construction", "contains", canonical)

    # Rocker profile — categorical column with a few canonical values.
    for kw, canonical in _SKI_ROCKER_KEYWORDS.items():
        idx = text.find(kw)
        if idx < 0:
            continue
        # "no rocker" is itself a negation cue but we WANT the resulting
        # camber filter as a positive constraint, so don't recurse the
        # negation check on rocker keywords whose canonical encodes the
        # "no" semantics already.
        is_inherently_neg = kw.startswith("no ")
        target = (
            filters if is_inherently_neg
            else (negative_filters if _is_negated(text, idx) else filters)
        )
        _add(target, "rocker_profile", "contains", canonical)

    # Brand match — products.brand column (text). The Rust prefilter joins
    # via product_attributes, but brand isn't an attribute — the
    # recommender exposes it through the brand attribute_def we add at
    # ingest time when present. If brand isn't in product_attributes the
    # SQL EXISTS check returns no rows so the prefilter falls back to the
    # full catalog (graceful degrade).
    for kw, canonical in _SKI_BRAND_KEYWORDS.items():
        idx = text.find(kw)
        if idx < 0:
            continue
        target = negative_filters if _is_negated(text, idx) else filters
        _add(target, "brand", "contains", canonical)

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
                _add(filters, attr, sub_op, val)
            range_attrs.add(attr)
            continue
        # Skip a strict-equality filter for an attribute where ANY other
        # numeric filter already fired — eq=90 contradicts lte=90 and
        # narrows the prefilter unnecessarily.
        if op == "eq" and any(
            seen_attr == attr and seen_op != "eq"
            for seen_attr, seen_op, _ in seen_keys
        ):
            continue
        try:
            val = int(m.group(1))
        except (IndexError, ValueError):
            continue
        _add(filters, attr, op, val)

    # Stiffness / damp / playfulness / etc. — keyword to scale-attribute rules.
    for pattern, op, attr, val in _SKI_STIFFNESS_RULES:
        if re.search(pattern, text):
            _add(filters, attr, op, val)

    return filters, negative_filters


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
# Generic catalog-derived hooks (Layer 1) + LLM-discovered phrases (Layer 2).
#
# The generic hooks let any new domain pick up price/points/range/category
# filters without hand-curating phrase tables. They run alongside the
# domain-specific handlers above; the domain handler covers the colorful
# vocabulary an LLM wouldn't infer ("ice coast", "japow"), generic covers
# the universals.
# ---------------------------------------------------------------------------


@dataclass
class _DomainSchema:
    """Auto-derived from `artifacts/<domain>/products.jsonl`.

    `numeric_attrs[name]` = (min, max) range observed in the catalog.
    `categorical_values[name][lowercased_value]` = canonical_value.
    `discovered_phrases` is loaded from `bundle/filter_phrases.json` if
    present (Layer 2 output).
    """
    numeric_attrs: dict[str, tuple[float, float]] = field(default_factory=dict)
    categorical_values: dict[str, dict[str, str]] = field(default_factory=dict)
    discovered_phrases: list[tuple[str, list[dict], bool]] = field(default_factory=list)


_PROJECT_ROOT_FOR_PARSER = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def _load_domain_schema(domain: str) -> _DomainSchema | None:
    """Read `artifacts/<domain>/products.jsonl` once and cache the schema."""
    if not domain:
        return None
    products_path = _PROJECT_ROOT_FOR_PARSER / "artifacts" / domain / "products.jsonl"
    if not products_path.exists():
        return _DomainSchema()  # empty but non-None: still apply phrase loader

    schema = _DomainSchema()
    numeric_seen: dict[str, list[float]] = {}
    cat_seen: dict[str, dict[str, str]] = {}
    for line in products_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue
        for top in ("attributes", "specs"):
            for k, v in (p.get(top) or {}).items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    numeric_seen.setdefault(k, []).append(float(v))
                elif isinstance(v, str) and v.strip():
                    cat = cat_seen.setdefault(k, {})
                    cat[v.lower()] = v
                elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                    cat = cat_seen.setdefault(k, {})
                    for x in v:
                        if x.strip():
                            cat[x.lower()] = x

    for k, vals in numeric_seen.items():
        schema.numeric_attrs[k] = (min(vals), max(vals))
    schema.categorical_values = cat_seen

    # Layer 2: discovered phrases.
    phrases_path = _PROJECT_ROOT_FOR_PARSER / "artifacts" / domain / "filter_phrases.json"
    if phrases_path.exists():
        try:
            data = json.loads(phrases_path.read_text())
            for entry in data.get("phrases", []):
                phrase = (entry.get("phrase") or "").lower().strip()
                hints = entry.get("filters") or []
                if not phrase or not hints:
                    continue
                # validate hint shape minimally
                clean = []
                for h in hints:
                    if all(k in h for k in ("attribute", "op", "value")):
                        clean.append({"attribute": h["attribute"], "op": h["op"], "value": h["value"]})
                if clean:
                    schema.discovered_phrases.append((phrase, clean, bool(entry.get("negated_only"))))
        except (json.JSONDecodeError, OSError):
            pass

    return schema


# Universal numeric patterns. Currency-aware; bare numbers are scoped to
# the attribute name they appear next to (e.g. "95 points") to avoid
# pulling random integers out of the query.
_NUMERIC_BARE = r"\$?(\d+(?:\.\d+)?)"
_NUMERIC_BARE_RANGE = r"\$?(\d+(?:\.\d+)?)\s*(?:-|to|–|—|and)\s*\$?(\d+(?:\.\d+)?)"


def _emit_range(filters: list[dict], attr: str, lo: float, hi: float) -> None:
    if hi < lo:
        lo, hi = hi, lo
    filters.append({"attribute": attr, "op": "gte", "value": _coerce(lo)})
    filters.append({"attribute": attr, "op": "lte", "value": _coerce(hi)})


def _coerce(v: float):
    return int(v) if float(v).is_integer() else v


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return s[idx]


def _apply_numeric_for_attr(
    text: str, attr: str, anchors: list[str],
    pos: list[dict], neg: list[dict],
) -> None:
    """Apply numeric extractors for one attribute. `anchors` lists tokens
    that scope a bare-number match (e.g. "points", "pts", "$" for price)."""
    # Range first so it doesn't get shadowed by single-bound matches.
    for anc in anchors:
        # Patterns like "between $X and $Y", "$X-$Y", "X to Y points"
        for pat in (
            rf"between\s+{_NUMERIC_BARE_RANGE}\s*{re.escape(anc)}",
            rf"{_NUMERIC_BARE_RANGE}\s*{re.escape(anc)}",
            rf"{re.escape(anc)}\s*{_NUMERIC_BARE_RANGE}",
        ):
            m = re.search(pat, text)
            if m:
                target = neg if _is_negated(text, m.start()) else pos
                _emit_range(target, attr, float(m.group(1)), float(m.group(2)))
                return

    # Single-bound: under/over/at least/at most/less than/more than.
    bounds = [
        (r"\bunder\s+", "lte"), (r"\bbelow\s+", "lte"),
        (r"\bless than\s+", "lte"), (r"\bat most\s+", "lte"),
        (r"\bover\s+", "gte"), (r"\babove\s+", "gte"),
        (r"\bmore than\s+", "gte"), (r"\bat least\s+", "gte"),
    ]
    for prefix, op in bounds:
        for anc in anchors:
            if anc == "$":
                # Currency: anchor leads the number, must be PRESENT to disambiguate
                # bare counts like "95 points".
                pat = rf"{prefix}{re.escape(anc)}(\d+(?:\.\d+)?)"
            else:
                pat = rf"{prefix}{_NUMERIC_BARE}\s*{re.escape(anc)}"
            m = re.search(pat, text)
            if not m:
                continue
            target = neg if _is_negated(text, m.start()) else pos
            target.append({"attribute": attr, "op": op, "value": _coerce(float(m.group(1)))})
            return

    # "X+ <anchor>" → gte
    for anc in anchors:
        m = re.search(rf"(\d+(?:\.\d+)?)\+\s*{re.escape(anc)}", text)
        if m:
            target = neg if _is_negated(text, m.start()) else pos
            target.append({"attribute": attr, "op": "gte", "value": _coerce(float(m.group(1)))})
            return


def _parse_generic(
    text: str, schema: _DomainSchema
) -> tuple[list[dict], list[dict]]:
    """Domain-agnostic extractors driven by the catalog schema."""
    pos: list[dict] = []
    neg: list[dict] = []

    # Numeric attributes — price gets currency anchor, points gets the
    # word "points"/"pts"/"point", anything else is opt-in via attr name.
    if "price" in schema.numeric_attrs:
        _apply_numeric_for_attr(text, "price", ["$", "dollars", "dollar", "usd"], pos, neg)
    if "points" in schema.numeric_attrs:
        _apply_numeric_for_attr(text, "points", ["points", "pts", "point", "pt"], pos, neg)
    for attr in schema.numeric_attrs:
        if attr in ("price", "points"):
            continue
        # Bare attribute-name anchor: e.g. "vintage 2010"
        _apply_numeric_for_attr(text, attr, [attr], pos, neg)

    # Quality-tier phrases mapped to numeric thresholds via catalog quantiles.
    if "price" in schema.numeric_attrs:
        # Recompute quantiles from the schema observation list — but we
        # only stored (min,max). Fall back to fixed-ish thresholds keyed
        # on the catalog's max so we adapt across domains.
        _, hi = schema.numeric_attrs["price"]
        cheap = max(15.0, hi * 0.10)
        premium = max(40.0, hi * 0.30)
        for phrase, op, val in (
            ("cheap", "lte", cheap), ("budget", "lte", cheap),
            ("affordable", "lte", cheap), ("inexpensive", "lte", cheap),
            ("expensive", "gte", premium), ("premium", "gte", premium),
            ("high-end", "gte", premium), ("luxury", "gte", premium),
        ):
            idx = text.find(phrase)
            if idx >= 0:
                target = neg if _is_negated(text, idx) else pos
                target.append({"attribute": "price", "op": op, "value": _coerce(val)})
                break  # one price-tier hint per query
    if "points" in schema.numeric_attrs:
        for phrase, op, val in (
            ("high-rated", "gte", 92), ("highly-rated", "gte", 92),
            ("highly rated", "gte", 92), ("top-rated", "gte", 94),
            ("well-rated", "gte", 90), ("well rated", "gte", 90),
            ("low-rated", "lte", 86), ("lowly-rated", "lte", 86),
        ):
            idx = text.find(phrase)
            if idx >= 0:
                target = neg if _is_negated(text, idx) else pos
                target.append({"attribute": "points", "op": op, "value": val})
                break

    # Categorical mentions: scan each text attr's value-set against the
    # query. Match longest-value-first so "cabernet sauvignon" beats
    # "cabernet". When the full value isn't in the query, fall back to
    # individual significant words (≥4 chars, not in stoplist) — that
    # lets "napa cabernet" hit Napa Valley + Cabernet Sauvignon.
    _CATEGORICAL_STOPWORDS = {
        "valley", "hills", "vineyard", "estate", "ridge", "creek", "river",
        "blend", "white", "red", "rose", "blanc", "noir", "the", "and", "of",
    }
    for attr, value_map in schema.categorical_values.items():
        if attr in ("taster",):
            continue
        if not value_map:
            continue
        # Pass 1: full-value match (longest first).
        full_matched = False
        for low_val in sorted(value_map.keys(), key=len, reverse=True):
            if len(low_val) < 3:
                continue
            idx = text.find(low_val)
            if idx < 0:
                continue
            pre_ok = idx == 0 or not text[idx - 1].isalnum()
            end = idx + len(low_val)
            post_ok = end == len(text) or not text[end].isalnum()
            if not (pre_ok and post_ok):
                continue
            target = neg if _is_negated(text, idx) else pos
            target.append({"attribute": attr, "op": "contains", "value": value_map[low_val]})
            full_matched = True
            break
        if full_matched:
            continue
        # Pass 2: word-level fallback — emit a contains filter on the matched
        # word itself so the Rust prefilter LIKE %word% picks up any
        # multi-word value containing it (e.g. "cabernet" → Cabernet Sauvignon
        # AND Cabernet Franc).
        word_index: set[str] = set()
        for low_val in value_map:
            for w in re.findall(r"[a-z][a-z\-]+", low_val):
                if len(w) >= 4 and w not in _CATEGORICAL_STOPWORDS:
                    word_index.add(w)
        for word in sorted(word_index, key=len, reverse=True):
            for m in re.finditer(rf"\b{re.escape(word)}\b", text):
                idx = m.start()
                target = neg if _is_negated(text, idx) else pos
                target.append({
                    "attribute": attr,
                    "op": "contains",
                    "value": word.capitalize(),
                })
                break

    return pos, neg


def _apply_discovered_phrases(
    text: str, phrases: list[tuple[str, list[dict], bool]],
) -> tuple[list[dict], list[dict]]:
    """Apply Layer 2 phrase mappings from `filter_phrases.json`."""
    pos: list[dict] = []
    neg: list[dict] = []
    for phrase, hints, negated_only in phrases:
        idx = text.find(phrase)
        if idx < 0:
            continue
        is_neg = negated_only or _is_negated(text, idx)
        target = neg if is_neg else pos
        for h in hints:
            target.append(dict(h))
    return pos, neg


def _merge_filters(a: list[dict], b: list[dict]) -> list[dict]:
    """Append filters from `b` to `a` skipping near-duplicates.

    For `contains` ops on text values, dedupe is case-insensitive AND
    treats prefix-overlap as a duplicate (so `titanal` and `Titanal` and
    `titanal_sandwich` all collapse to one filter). The first-seen value
    wins.
    """
    out = list(a)
    seen_keys: set[tuple[str, str, object]] = set()
    seen_text_prefixes: dict[tuple[str, str], list[str]] = {}
    for f in out:
        key = (f["attribute"], f["op"], _hashable(f["value"]))
        seen_keys.add(key)
        if f["op"] in ("contains", "not_contains") and isinstance(f["value"], str):
            seen_text_prefixes.setdefault((f["attribute"], f["op"]), []).append(f["value"].lower())
    for f in b:
        key = (f["attribute"], f["op"], _hashable(f["value"]))
        if key in seen_keys:
            continue
        if f["op"] in ("contains", "not_contains") and isinstance(f["value"], str):
            low = f["value"].lower()
            existing = seen_text_prefixes.get((f["attribute"], f["op"]), [])
            if any(e == low or e.startswith(low) or low.startswith(e) for e in existing):
                continue
            seen_text_prefixes.setdefault((f["attribute"], f["op"]), []).append(low)
        seen_keys.add(key)
        out.append(f)
    return out


def _hashable(v):
    if isinstance(v, list):
        return tuple(v)
    return v


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
