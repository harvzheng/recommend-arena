"""LLM-based query parsing for Design 05: SQL-First / SQLite + FTS5.

Parses a natural language query into structured filters and keywords
using a single LLM call. This is the ONLY LLM call at query time.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from .synonyms import pre_expand_query

logger = logging.getLogger(__name__)

QUERY_PARSE_PROMPT = """Parse this product query into structured filters and free-text keywords.
Domain: {domain}

Available attributes and their types:
{attribute_defs_description}

Query: "{user_query}"

Return JSON with this structure:
{{
  "filters": [
    {{"attribute": "attribute_name", "op": "gte|lte|eq|contains|not_contains", "value": <number_or_string>}}
  ],
  "keywords": "free text search terms not captured by filters"
}}

Rules:
- For scale attributes (1-10), use numeric ops (gte, lte, eq)
- For categorical attributes, use "contains" or "not_contains"
- "stiff" means stiffness >= 7, "very stiff" means stiffness >= 8
- "soft" or "buttery" means stiffness <= 4, "very soft" means stiffness <= 3
- "forgiving" means forgiveness >= 6 and stiffness <= 5
- "180cm+" means look for lengths_available containing values >= 180
- "under 90mm waist" means waist_width_mm < 90, use lte with value 89
- "NOT playful" or "not playful" means playfulness <= 4
- Width ranges like "96-100mm waist" become two filters: waist_width_mm gte 96, waist_width_mm lte 100
- "lightweight" or "light" for skis means weight_g lte 1800
- "damp" or "dampness" maps to the "damp" attribute, damp gte 6 or 7
- "ice coast" implies edge_grip gte 7, damp gte 6 — the ski needs to hold on hardpack/ice
- "confidence-inspiring" implies stability_at_speed gte 7 and damp gte 6
- "charger" implies stiffness gte 7, damp gte 7, stability_at_speed gte 7
- "surfy" implies powder_float gte 7 and playfulness gte 6
- "poppy" or "alive underfoot" implies playfulness gte 7, responsiveness gte 7
- "bouncy" for shoes implies responsiveness gte 7 and cushioning gte 6
- "plush" for shoes implies cushioning gte 7
- "nimble" for shoes implies responsiveness gte 6, weight_feel gte 7
- Put subjective/vibes-based terms in keywords AND also map them to filters
- Put domain-specific jargon in keywords AND also map them to filters
- If the query mentions terrain types, add them as terrain contains filters
- Always include at least some keywords for FTS matching

Examples:

Query (ski): "Looking for an ice coast charger that holds an edge on hardpack, available in 180cm+"
Answer:
{{"filters": [{{"attribute": "edge_grip", "op": "gte", "value": 7}}, {{"attribute": "damp", "op": "gte", "value": 7}}, {{"attribute": "stability_at_speed", "op": "gte", "value": 7}}, {{"attribute": "stiffness", "op": "gte", "value": 7}}, {{"attribute": "lengths_available", "op": "gte", "value": 180}}], "keywords": "ice coast charger hardpack edge grip damp stable"}}

Query (ski): "Something buttery and surfy for powder days, NOT a charger"
Answer:
{{"filters": [{{"attribute": "stiffness", "op": "lte", "value": 4}}, {{"attribute": "playfulness", "op": "gte", "value": 6}}, {{"attribute": "powder_float", "op": "gte", "value": 7}}, {{"attribute": "stability_at_speed", "op": "lte", "value": 5}}], "keywords": "buttery surfy powder float playful soft"}}

Query (running_shoe): "Need a bouncy daily trainer that can handle speed work but won't fall apart"
Answer:
{{"filters": [{{"attribute": "responsiveness", "op": "gte", "value": 7}}, {{"attribute": "cushioning", "op": "gte", "value": 6}}, {{"attribute": "durability", "op": "gte", "value": 6}}], "keywords": "bouncy responsive daily trainer speed durable"}}

Return ONLY valid JSON, no other text."""


def get_attribute_defs_description(db: sqlite3.Connection, domain_id: int) -> str:
    """Build a human-readable description of available attributes."""
    rows = db.execute(
        "SELECT name, data_type, scale_min, scale_max, allowed_values "
        "FROM attribute_defs WHERE domain_id = ? ORDER BY name",
        (domain_id,),
    ).fetchall()

    lines = []
    for name, dtype, smin, smax, allowed in rows:
        desc = f"- {name} ({dtype})"
        if dtype == "scale" and smin is not None and smax is not None:
            desc += f" range {smin:.0f}-{smax:.0f}"
        if allowed:
            desc += f" values: {allowed}"
        lines.append(desc)
    return "\n".join(lines)


def parse_query(
    llm_provider,
    db: sqlite3.Connection,
    user_query: str,
    domain: str,
    domain_id: int,
) -> dict:
    """Parse a natural language query into structured filters + keywords.

    Returns a dict with:
        - filters: list of {attribute, op, value}
        - keywords: str of free-text keywords for FTS
    """
    # Pre-expand query using phrase dictionary before LLM parsing
    expanded_query, extra_filters = pre_expand_query(user_query, domain)

    attr_desc = get_attribute_defs_description(db, domain_id)

    prompt = QUERY_PARSE_PROMPT.format(
        domain=domain,
        attribute_defs_description=attr_desc,
        user_query=expanded_query,
    )

    try:
        response = llm_provider.generate(prompt, json_mode=True)
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            response = "\n".join(lines)
        parsed = json.loads(response)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to parse query with LLM: %s", e)
        # Fallback: treat entire query as keywords
        parsed = {"filters": [], "keywords": user_query}

    # Validate and clean up filters
    if "filters" not in parsed:
        parsed["filters"] = []
    if "keywords" not in parsed:
        parsed["keywords"] = user_query

    valid_filters = []
    for f in parsed["filters"]:
        if not isinstance(f, dict):
            continue
        if "attribute" not in f or "op" not in f or "value" not in f:
            continue
        if f["op"] not in ("gte", "lte", "eq", "contains", "not_contains"):
            continue
        valid_filters.append(f)

    # Merge extra filters from phrase expansion (only if LLM didn't already cover them)
    existing_attrs = {(f["attribute"], f["op"]) for f in valid_filters}
    for ef in extra_filters:
        key = (ef["attribute"], ef["op"])
        if key not in existing_attrs:
            valid_filters.append(ef)
            existing_attrs.add(key)

    parsed["filters"] = valid_filters
    return parsed
