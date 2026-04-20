"""LLM-based facet extraction from reviews."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def extract_facets(
    review_text: str,
    domain_config: dict,
    llm,
) -> dict:
    """Extract structured facets from a single review using the LLM.

    Args:
        review_text: Raw review text.
        domain_config: Domain configuration with facet definitions.
        llm: LLMProvider instance.

    Returns:
        Dict mapping facet names to extracted values.
    """
    facets = domain_config["facets"]
    facet_descriptions = {}
    for name, fdef in facets.items():
        desc = fdef.get("description", name)
        ftype = fdef["type"]
        if ftype == "numeric":
            facet_descriptions[name] = f"{desc} (score 0.0 to 1.0)"
        elif ftype == "categorical":
            facet_descriptions[name] = f"{desc} (pick from: {fdef['values']})"
        elif ftype == "boolean":
            facet_descriptions[name] = f"{desc} (true/false)"
        elif ftype == "spec_numeric":
            facet_descriptions[name] = f"{desc} ({fdef.get('unit', '')})"

    prompt = f"""Extract product attributes from this review.

Attributes to look for:
{json.dumps(facet_descriptions, indent=2)}

For numeric sentiment attributes (type numeric), score 0.0 (very low/negative) to 1.0 (very high/positive).
For categorical attributes, pick from the allowed values only.
For boolean attributes, return true or false.
For spec_numeric attributes, return the numeric value if mentioned.
Only include attributes explicitly discussed or clearly implied in the review.
Return ONLY valid JSON with attribute names as keys.

Review: "{review_text}"
"""

    try:
        raw = llm.generate(prompt, json_mode=True)
        parsed = _parse_json_response(raw)
        return _validate_extracted(parsed, domain_config)
    except Exception as e:
        logger.warning("Facet extraction failed: %s", e)
        return {}


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    return json.loads(raw)


def _validate_extracted(parsed: dict, domain_config: dict) -> dict:
    """Validate and clean extracted facet values against the schema."""
    facets = domain_config["facets"]
    validated = {}

    for name, value in parsed.items():
        if name not in facets:
            continue
        fdef = facets[name]
        ftype = fdef["type"]

        try:
            if ftype == "numeric":
                v = float(value)
                validated[name] = max(0.0, min(1.0, v))
            elif ftype == "categorical":
                allowed = fdef.get("values", [])
                if isinstance(value, list):
                    validated[name] = [v for v in value if v in allowed]
                elif value in allowed:
                    validated[name] = [value]
            elif ftype == "boolean":
                if isinstance(value, bool):
                    validated[name] = value
                elif isinstance(value, str):
                    validated[name] = value.lower() in ("true", "yes", "1")
            elif ftype == "spec_numeric":
                validated[name] = float(value)
        except (ValueError, TypeError):
            continue

    return validated
