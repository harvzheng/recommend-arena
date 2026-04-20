"""LLM-based natural language query parser.

Translates natural language queries into structured facet filters
and text search terms.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def parse_query(
    nl_query: str,
    domain_config: dict,
    llm,
) -> dict:
    """Parse a natural language query into structured search parameters.

    Args:
        nl_query: Natural language query string.
        domain_config: Domain configuration with facet definitions.
        llm: LLMProvider instance.

    Returns:
        Dict with keys:
            - text_query: str (for FTS)
            - filters: dict of {field: (operator, value)}
            - sort_fields: list of (field, direction)
            - negations: list of negation descriptions
    """
    facets = domain_config["facets"]

    # Build a concise facet description for the prompt
    facet_info = {}
    for name, fdef in facets.items():
        ftype = fdef["type"]
        desc = fdef.get("description", name)
        synonyms = fdef.get("synonyms", [])
        info = {"type": ftype, "description": desc}
        if synonyms:
            info["synonyms"] = synonyms
        if ftype == "categorical":
            info["values"] = fdef.get("values", [])
        if ftype == "numeric":
            info["range"] = fdef.get("range", [0.0, 1.0])
        facet_info[name] = info

    prompt = f"""Convert this product search query into structured search parameters.

Query: "{nl_query}"

Available facets:
{json.dumps(facet_info, indent=2)}

Return JSON with these keys:
- "text_query": string of subjective/vague terms for full-text search (or "*" if all terms map to facets)
- "filters": object mapping facet names to filter conditions, each as {{"op": ">"|"<"|">="|"<="|"=", "value": ...}}
  - For numeric facets (0.0-1.0 range): use thresholds like {{"op": ">", "value": 0.7}} for "high/stiff/strong" or {{"op": "<", "value": 0.3}} for "low/soft/weak"
  - For categorical facets: use {{"op": "=", "value": "category_name"}}
  - For boolean facets: use {{"op": "=", "value": true/false}}
  - For spec_numeric facets: use appropriate comparison {{"op": ">=", "value": 180}}
- "sort_fields": list of {{"field": "name", "direction": "desc"}} for explicit sorting priorities
- "negations": list of strings describing any "not X" / "without X" conditions (already encoded in filters above)

Important rules:
- Map subjective terms (stiff, soft, damp, playful, cushy, etc.) to numeric facet filters
- "stiff" -> stiffness > 0.7, "very stiff" -> stiffness > 0.85, "soft" -> stiffness < 0.3
- "NOT playful" or "not playful" -> playfulness < 0.4
- Keep text_query for terms that don't map to any facet (brand names, specific phrases, subjective feelings)
- For waist width queries like "95mm waist" or "under 90mm", use waist_width_mm
- For length queries like "180cm+" or "available in 170-175cm", use length_cm
- If the query mentions a specific construction or material, put it in text_query
"""

    try:
        raw = llm.generate(prompt, json_mode=True)
        parsed = _parse_json_response(raw)
        return _validate_and_clean(parsed, domain_config, nl_query)
    except Exception as e:
        logger.warning("Query parsing failed: %s — using text-only search", e)
        return {
            "text_query": nl_query,
            "filters": {},
            "sort_fields": [],
            "negations": [],
        }


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from LLM response."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    return json.loads(raw)


def _validate_and_clean(
    parsed: dict,
    domain_config: dict,
    original_query: str,
) -> dict:
    """Validate LLM-generated filter syntax; drop invalid fields."""
    facets = domain_config["facets"]
    valid_fields = set(facets.keys())

    # Normalize filters
    raw_filters = parsed.get("filters", {})
    clean_filters: dict[str, tuple[str, str]] = {}

    if isinstance(raw_filters, dict):
        for field, condition in raw_filters.items():
            if field not in valid_fields:
                continue
            fdef = facets[field]
            ftype = fdef["type"]

            if isinstance(condition, dict):
                op = condition.get("op", "=")
                value = condition.get("value")
            elif isinstance(condition, str):
                # Try parsing "op value" format
                m = re.match(r"([><=!]+)\s*(.+)", condition)
                if m:
                    op, value = m.group(1), m.group(2)
                else:
                    op, value = "=", condition
            else:
                op, value = "=", condition

            # Validate by type
            if ftype in ("numeric", "spec_numeric"):
                try:
                    value = str(float(value))
                    clean_filters[field] = (op, value)
                except (ValueError, TypeError):
                    pass
            elif ftype == "categorical":
                allowed = fdef.get("values", [])
                str_val = str(value)
                if str_val in allowed:
                    clean_filters[field] = ("=", str_val)
            elif ftype == "boolean":
                if isinstance(value, bool):
                    clean_filters[field] = ("=", str(value).lower())
                elif isinstance(value, str):
                    clean_filters[field] = ("=", value.lower())

    # Normalize sort fields
    raw_sorts = parsed.get("sort_fields", [])
    sort_fields = []
    if isinstance(raw_sorts, list):
        for s in raw_sorts:
            if isinstance(s, dict):
                field = s.get("field", "")
                direction = s.get("direction", "desc")
                if field in valid_fields or field in ("popularity", "review_count"):
                    sort_fields.append((field, direction))
            elif isinstance(s, str):
                parts = s.split(":")
                if len(parts) == 2 and parts[0] in valid_fields:
                    sort_fields.append((parts[0], parts[1]))

    # Get text query
    text_query = parsed.get("text_query", original_query)
    if not text_query or text_query == "":
        text_query = "*"

    return {
        "text_query": text_query,
        "filters": clean_filters,
        "sort_fields": sort_fields,
        "negations": parsed.get("negations", []),
    }
