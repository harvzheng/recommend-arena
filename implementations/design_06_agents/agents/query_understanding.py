"""Query Understanding Agent -- parses natural language queries into structured form.

This is the key LLM-powered step in the query pipeline. It identifies hard filters
(must-match constraints) and soft preferences (weighted desires) from the user's
natural language query.
"""

from __future__ import annotations

import json
import logging
import re

from ..state import ParsedQuery

logger = logging.getLogger(__name__)

# Domain-specific attribute schemas for the LLM prompt
DOMAIN_SCHEMAS = {
    "ski": {
        "numeric_attributes": {
            "stiffness": "1-10, how stiff/demanding the ski is (10 = race stiff)",
            "damp": "1-10, vibration absorption (10 = extremely damp)",
            "edge_grip": "1-10, grip on hard snow/ice (10 = race grip)",
            "stability_at_speed": "1-10, composure at high speed (10 = rock solid)",
            "playfulness": "1-10, fun/agility/pop (10 = very playful)",
            "powder_float": "1-10, flotation in deep snow (10 = max float)",
            "forgiveness": "1-10, tolerance for imperfect technique (10 = very forgiving)",
        },
        "spec_filters": {
            "waist_width_mm": "waist width in mm (e.g. 88, 100)",
            "turn_radius_m": "turn radius in meters",
            "weight_g_per_ski": "weight per ski in grams",
            "lengths_cm": "available lengths in cm (list)",
            "rocker_profile": "camber | rocker_camber | rocker_camber_rocker | full_rocker",
            "construction": "construction type",
        },
        "terrain_values": ["on-piste", "off-piste", "all-mountain", "park", "race",
                           "carving", "freeride", "backcountry", "groomed"],
        "synonyms": {
            "groomer": "on-piste", "groomers": "on-piste",
            "powder": "off-piste", "backcountry": "off-piste",
            "ice coast": "edge_grip + damp", "east coast": "edge_grip + damp",
            "noodly": "low stiffness", "flexy": "low stiffness",
            "charger": "high stability_at_speed + high stiffness",
        },
    },
    "running_shoe": {
        "numeric_attributes": {
            "cushioning": "1-10, amount of cushion/protection",
            "responsiveness": "1-10, energy return and snappiness",
            "stability": "1-10, support and control",
            "grip": "1-10, traction/outsole grip",
            "breathability": "1-10, ventilation",
            "durability": "1-10, longevity",
            "weight_feel": "1-10, perceived lightness (10 = very light)",
        },
        "spec_filters": {
            "weight_g": "shoe weight in grams",
            "heel_drop_mm": "heel-to-toe drop in mm",
            "stack_height_mm": "stack height in mm",
            "surface": "road | trail",
        },
        "terrain_values": ["road", "trail"],
        "synonyms": {
            "marathon": "high cushioning + long distance",
            "racing": "high responsiveness + low weight",
            "daily trainer": "balanced cushioning + durability",
        },
    },
}

QUERY_PARSE_PROMPT = """You are a {domain} product recommendation system. Parse this user query into structured filters and preferences.

User query: "{query}"

Available numeric attributes (score 1-10):
{attributes_desc}

Available spec filters:
{specs_desc}

Common synonyms: {synonyms}

Return ONLY valid JSON with this structure (no markdown, no commentary):
{{
  "hard_filters": {{
    "spec_name": value_or_range_object
  }},
  "soft_preferences": {{
    "attribute_name": preference_value_1_to_10
  }},
  "negative_preferences": {{
    "attribute_name": preference_value_1_to_10
  }},
  "terrain": ["terrain_value"]
}}

Rules:
- hard_filters are MUST-MATCH constraints from the query (exact values or ranges like {{"min": 180}}).
  Use range objects for size/width/weight constraints. Example: "lengths_cm": {{"min": 180}} or "waist_width_mm": {{"min": 95, "max": 100}}
- soft_preferences are DESIRED attributes with how strongly desired (1-10).
  Higher means user wants MORE of that attribute.
- negative_preferences are attributes the user explicitly does NOT want (1-10 for strength of avoidance).
- terrain is a list of terrain types mentioned or implied.
- Only include attributes/filters actually mentioned or strongly implied by the query.
- For "stiff" -> soft_preferences: {{"stiffness": 9}}
- For "beginner" -> soft_preferences: {{"forgiveness": 9}}, negative_preferences: {{"stiffness": 7}}
- For "NOT playful" -> negative_preferences: {{"playfulness": 8}}"""


def query_understanding_agent(state: dict, llm_provider=None) -> dict:
    """Parse a raw query into a structured ParsedQuery.

    Args:
        state: Pipeline state dict with "raw_query" and "domain".
        llm_provider: Shared LLM provider.

    Returns:
        Updated state with "parsed_query" set.
    """
    raw_query = state["raw_query"]
    domain = state.get("domain", "ski")

    if llm_provider is not None:
        parsed = _parse_with_llm(raw_query, domain, llm_provider)
        if parsed is not None:
            return {**state, "parsed_query": parsed}

    # Fallback: keyword-based parsing
    logger.info("Using keyword fallback for query: %s", raw_query)
    fallback = _parse_with_keywords(raw_query, domain)
    errors = state.get("errors", []) + [
        "Query understanding fell back to keyword parsing"
    ] if llm_provider is not None else state.get("errors", [])

    return {**state, "parsed_query": fallback, "errors": errors}


def _parse_with_llm(
    query: str, domain: str, llm_provider
) -> ParsedQuery | None:
    """Try to parse query using LLM with retries."""
    schema = DOMAIN_SCHEMAS.get(domain)
    if schema is None:
        return None

    attrs_desc = "\n".join(
        f"  - {k}: {v}" for k, v in schema["numeric_attributes"].items()
    )
    specs_desc = "\n".join(
        f"  - {k}: {v}" for k, v in schema["spec_filters"].items()
    )
    synonyms = ", ".join(
        f'"{k}" = {v}' for k, v in schema.get("synonyms", {}).items()
    )

    prompt = QUERY_PARSE_PROMPT.format(
        domain=domain,
        query=query,
        attributes_desc=attrs_desc,
        specs_desc=specs_desc,
        synonyms=synonyms,
    )

    max_retries = 2
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = llm_provider.generate(prompt, json_mode=True)
            cleaned = _clean_json_response(response)
            parsed = json.loads(cleaned)

            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict, got {type(parsed)}")

            return ParsedQuery(
                domain=domain,
                hard_filters=parsed.get("hard_filters", {}),
                soft_preferences=_normalize_preferences(
                    parsed.get("soft_preferences", {})
                ),
                negative_preferences=_normalize_preferences(
                    parsed.get("negative_preferences", {})
                ),
                freetext_intent=query,
            )

        except (json.JSONDecodeError, ValueError, KeyError, RuntimeError) as e:
            last_error = str(e)
            logger.warning(
                "Query parse attempt %d failed: %s", attempt + 1, last_error
            )
            if isinstance(e, RuntimeError):
                # Provider is down, no point retrying
                break

    logger.warning("All query parse attempts failed: %s", last_error)
    return None


def _parse_with_keywords(query: str, domain: str) -> ParsedQuery:
    """Simple keyword-based query parsing as fallback."""
    query_lower = query.lower()
    soft_prefs: dict[str, float] = {}
    neg_prefs: dict[str, float] = {}
    hard_filters: dict[str, any] = {}

    if domain == "ski":
        # Stiffness
        if "stiff" in query_lower:
            soft_prefs["stiffness"] = 9.0
        if "soft" in query_lower or "flexible" in query_lower:
            soft_prefs["stiffness"] = 2.0
            soft_prefs["forgiveness"] = 8.0

        # Terrain
        if "powder" in query_lower:
            soft_prefs["powder_float"] = 9.0
        if "carv" in query_lower or "on-piste" in query_lower or "groomer" in query_lower:
            soft_prefs["edge_grip"] = 8.0
        if "park" in query_lower or "freestyle" in query_lower:
            soft_prefs["playfulness"] = 9.0
        if "all-mountain" in query_lower or "all mountain" in query_lower:
            soft_prefs["stability_at_speed"] = 6.0
            soft_prefs["powder_float"] = 5.0
            soft_prefs["edge_grip"] = 6.0

        # Performance
        if "damp" in query_lower:
            soft_prefs["damp"] = 8.0
        if "stable" in query_lower or "stability" in query_lower:
            soft_prefs["stability_at_speed"] = 8.0
        if "playful" in query_lower and "not playful" not in query_lower:
            soft_prefs["playfulness"] = 8.0
        if "not playful" in query_lower:
            neg_prefs["playfulness"] = 8.0
        if "forgiv" in query_lower or "beginner" in query_lower:
            soft_prefs["forgiveness"] = 8.0
        if "edge" in query_lower or "grip" in query_lower:
            soft_prefs["edge_grip"] = 8.0
        if "float" in query_lower:
            soft_prefs["powder_float"] = 8.0
        if "light" in query_lower:
            soft_prefs.setdefault("playfulness", 6.0)

        # Freeride
        if "freeride" in query_lower:
            soft_prefs["powder_float"] = 7.0
            soft_prefs.setdefault("stability_at_speed", 7.0)

        # Ice coast
        if "ice coast" in query_lower or "east coast" in query_lower:
            soft_prefs["edge_grip"] = 9.0
            soft_prefs["damp"] = 8.0

        # Confidence
        if "confidence" in query_lower or "safe" in query_lower:
            soft_prefs["stability_at_speed"] = 8.0
            soft_prefs["damp"] = 7.0

        # Length filters
        length_match = re.search(r"(\d{3})\s*cm", query_lower)
        if length_match:
            length = int(length_match.group(1))
            if "+" in query_lower or "longer" in query_lower or "or longer" in query_lower:
                hard_filters["lengths_cm"] = {"min": length}
            elif "under" in query_lower or "shorter" in query_lower:
                hard_filters["lengths_cm"] = {"max": length}
            else:
                hard_filters["lengths_cm"] = {"min": length - 3, "max": length + 3}

        # Waist width
        waist_match = re.search(r"(\d{2,3})\s*mm\s*waist", query_lower)
        if waist_match:
            waist = int(waist_match.group(1))
            hard_filters["waist_width_mm"] = {"min": waist - 5, "max": waist + 5}

        # Waist range in query
        waist_range = re.search(
            r"(\d{2,3})\s*-\s*(\d{2,3})\s*mm\s*waist", query_lower
        )
        if waist_range:
            hard_filters["waist_width_mm"] = {
                "min": int(waist_range.group(1)),
                "max": int(waist_range.group(2)),
            }

        # Construction
        if "titanal" in query_lower:
            hard_filters["construction"] = "titanal_sandwich"
        if "camber" in query_lower and "rocker" not in query_lower:
            hard_filters["rocker_profile"] = "camber"

    elif domain == "running_shoe":
        if "cushion" in query_lower:
            soft_prefs["cushioning"] = 9.0
        if "responsive" in query_lower or "fast" in query_lower:
            soft_prefs["responsiveness"] = 8.0
        if "stable" in query_lower or "stability" in query_lower:
            soft_prefs["stability"] = 8.0
        if "grip" in query_lower or "traction" in query_lower:
            soft_prefs["grip"] = 8.0
        if "light" in query_lower:
            soft_prefs["weight_feel"] = 8.0
        if "durable" in query_lower or "durability" in query_lower:
            soft_prefs["durability"] = 8.0
        if "breathab" in query_lower:
            soft_prefs["breathability"] = 8.0
        if "trail" in query_lower:
            hard_filters["surface"] = "trail"
        if "road" in query_lower:
            hard_filters["surface"] = "road"
        if "marathon" in query_lower or "long" in query_lower:
            soft_prefs.setdefault("cushioning", 8.0)
            soft_prefs.setdefault("durability", 7.0)
        if "race" in query_lower or "racing" in query_lower:
            soft_prefs.setdefault("responsiveness", 9.0)
            soft_prefs.setdefault("weight_feel", 9.0)

    return ParsedQuery(
        domain=domain,
        hard_filters=hard_filters,
        soft_preferences=soft_prefs,
        negative_preferences=neg_prefs,
        freetext_intent=query,
    )


def _normalize_preferences(prefs: dict) -> dict[str, float]:
    """Normalize preference values to floats."""
    result = {}
    for k, v in prefs.items():
        try:
            result[k] = float(v)
        except (TypeError, ValueError):
            pass
    return result


def _clean_json_response(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
