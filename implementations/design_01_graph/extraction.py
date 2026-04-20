"""ABSA (Aspect-Based Sentiment Analysis) extraction using LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.llm_provider import LLMProvider

from .config import DOMAIN_SYNONYMS, get_domain_config

logger = logging.getLogger(__name__)


def _build_extraction_prompt(text: str, domain: str, known_attributes: list[str]) -> str:
    """Build the LLM prompt for aspect-sentiment extraction."""
    attr_list = ", ".join(known_attributes) if known_attributes else "any relevant product attributes"
    return f"""Analyze this {domain} product review and extract aspect-sentiment pairs.

For each product attribute mentioned in the review, return:
- "attribute": the canonical attribute name (use one of: {attr_list}, or a new descriptive name)
- "sentiment": a score from -1.0 (very negative) to +1.0 (very positive)
- "snippet": the key phrase from the review supporting this assessment (max 30 words)

Review: "{text}"

Return ONLY a JSON object with an "aspects" key containing an array of objects.
Example: {{"aspects": [{{"attribute": "stiffness", "sentiment": 0.9, "snippet": "incredibly stiff and powerful"}}]}}"""


def _parse_extraction_response(response: str) -> list[dict]:
    """Parse the LLM response into structured aspect-sentiment pairs."""
    # Try to extract JSON from the response
    try:
        # Try direct parse first
        data = json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM extraction response")
                return []
        else:
            # Try to find a JSON array
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                try:
                    aspects = json.loads(match.group())
                    if isinstance(aspects, list):
                        return [
                            a for a in aspects
                            if isinstance(a, dict) and "attribute" in a and "sentiment" in a
                        ]
                except json.JSONDecodeError:
                    pass
            logger.warning("No JSON found in LLM extraction response")
            return []

    # Handle different response shapes
    if isinstance(data, dict):
        aspects = data.get("aspects", data.get("results", data.get("pairs", [])))
        if isinstance(aspects, list):
            return [
                a for a in aspects
                if isinstance(a, dict) and "attribute" in a and "sentiment" in a
            ]
        # Maybe the dict itself is a single aspect
        if "attribute" in data and "sentiment" in data:
            return [data]
    elif isinstance(data, list):
        return [
            a for a in data
            if isinstance(a, dict) and "attribute" in a and "sentiment" in a
        ]

    return []


def _normalize_attribute_name(raw_name: str, domain_config: dict) -> str | None:
    """Map a raw attribute name to a canonical attribute name using synonyms."""
    raw_lower = raw_name.lower().strip().replace("_", " ")
    attributes = domain_config.get("attributes", {})

    # Direct match
    if raw_lower in attributes:
        return raw_lower

    # Synonym match
    for canonical, attr_info in attributes.items():
        synonyms = [s.lower() for s in attr_info.get("synonyms", [])]
        if raw_lower in synonyms:
            return canonical
        # Partial match: check if the raw name contains or is contained by a synonym
        for syn in synonyms:
            if raw_lower in syn or syn in raw_lower:
                return canonical

    # No match found — return the raw name normalized (allows discovery of new attributes)
    return raw_lower.replace(" ", "_")


def extract_aspects_from_review(
    text: str,
    domain: str,
    llm: LLMProvider,
) -> list[dict]:
    """Extract aspect-sentiment pairs from a single review.

    Returns a list of dicts, each with:
        - attribute: canonical attribute name
        - sentiment: float in [-1, 1]
        - snippet: supporting text from the review
    """
    domain_config = get_domain_config(domain)
    known_attributes = list(domain_config.get("attributes", {}).keys())
    prompt = _build_extraction_prompt(text, domain, known_attributes)

    try:
        response = llm.generate(prompt, json_mode=True)
    except Exception as e:
        logger.warning("LLM extraction failed: %s", e)
        return []

    raw_aspects = _parse_extraction_response(response)

    # Normalize attribute names and clamp sentiments
    results = []
    for aspect in raw_aspects:
        canonical = _normalize_attribute_name(aspect["attribute"], domain_config)
        if canonical is None:
            continue
        sentiment = float(aspect["sentiment"])
        sentiment = max(-1.0, min(1.0, sentiment))
        snippet = aspect.get("snippet", "")
        results.append({
            "attribute": canonical,
            "sentiment": sentiment,
            "snippet": str(snippet)[:200],
        })

    return results


def expand_synonyms(query_text: str, domain: str) -> list[dict]:
    """Expand colloquial terms in a query into structured attribute targets.

    Returns a list of {attribute, polarity, weight} dicts for any matched
    synonym phrases. These supplement (not replace) the LLM-parsed attributes.
    """
    query_lower = query_text.lower()
    domain_syns = DOMAIN_SYNONYMS.get(domain, {})
    expanded = []
    seen = set()

    # Sort by phrase length descending so longer matches take priority
    for phrase in sorted(domain_syns, key=len, reverse=True):
        if phrase in query_lower:
            for attr_name, polarity, weight in domain_syns[phrase]:
                if attr_name not in seen:
                    expanded.append({
                        "attribute": attr_name,
                        "polarity": polarity,
                        "weight": weight,
                    })
                    seen.add(attr_name)

    return expanded


def extract_query_attributes(
    query_text: str,
    domain: str,
    llm: LLMProvider,
) -> dict:
    """Parse a user query into structured attribute targets and constraints.

    Returns a dict with:
        - attributes: list of {attribute, polarity, weight}
        - categories: list of category strings
        - constraints: list of {field, op, value}
    """
    domain_config = get_domain_config(domain)
    known_attributes = list(domain_config.get("attributes", {}).keys())
    known_categories = domain_config.get("categories", [])
    category_synonyms = domain_config.get("category_synonyms", {})

    # Flatten category synonyms for the prompt
    cat_examples = []
    for cat, syns in category_synonyms.items():
        cat_examples.append(f"{cat} (matches: {', '.join(syns[:3])})")

    prompt = f"""Analyze this {domain} product search query and extract the user's preferences.

Query: "{query_text}"

Known attributes (rate desire on scale of -1 to +1, where +1 means they want a lot of it, -1 means they want very little):
{', '.join(known_attributes)}

Known categories:
{chr(10).join(cat_examples)}

Return a JSON object with:
- "attributes": array of {{"attribute": "name", "polarity": float (-1 to +1), "weight": float (0.5-2.0, higher = more important)}}
- "categories": array of matching category strings from the known list
- "constraints": array of {{"field": "spec_name", "op": ">=" or "<=" or "==", "value": number_or_string}}

Only include attributes that are clearly mentioned or implied in the query.
For constraints, look for things like "180cm+" (lengths_cm >= 180), "under 1500g" (weight_g_per_ski <= 1500), etc.

### Few-shot Examples ###

Query: "Looking for a stiff charger that can handle ice coast conditions, 180cm+"
Result: {{"attributes": [{{"attribute": "stiffness", "polarity": 1.0, "weight": 1.5}}, {{"attribute": "damp", "polarity": 1.0, "weight": 1.2}}, {{"attribute": "edge_grip", "polarity": 1.0, "weight": 1.5}}, {{"attribute": "stability_at_speed", "polarity": 1.0, "weight": 1.5}}], "categories": ["expert_carving", "advanced_frontside"], "constraints": [{{"field": "lengths_cm", "op": ">=", "value": 180}}]}}

Query: "Something playful and forgiving, NOT a stiff race ski"
Result: {{"attributes": [{{"attribute": "playfulness", "polarity": 1.0, "weight": 1.5}}, {{"attribute": "forgiveness", "polarity": 1.0, "weight": 1.5}}, {{"attribute": "stiffness", "polarity": -1.0, "weight": 1.5}}], "categories": ["all_mountain"], "constraints": []}}

Query: "Versatile all-mountain ski that can handle powder days but still carve groomers"
Result: {{"attributes": [{{"attribute": "versatility", "polarity": 1.0, "weight": 1.5}}, {{"attribute": "powder_float", "polarity": 1.0, "weight": 1.2}}, {{"attribute": "edge_grip", "polarity": 1.0, "weight": 1.0}}, {{"attribute": "playfulness", "polarity": 1.0, "weight": 0.8}}], "categories": ["all_mountain"], "constraints": []}}

### End Examples ###

Now analyze the query above and return ONLY the JSON result."""

    try:
        response = llm.generate(prompt, json_mode=True)
    except Exception as e:
        logger.warning("LLM query parsing failed: %s — falling back to keyword matching", e)
        return _fallback_query_parse(query_text, domain_config)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return _fallback_query_parse(query_text, domain_config)
        else:
            return _fallback_query_parse(query_text, domain_config)

    # Normalize parsed query attributes
    result = {"attributes": [], "categories": [], "constraints": []}

    for attr in data.get("attributes", []):
        canonical = _normalize_attribute_name(attr.get("attribute", ""), domain_config)
        if canonical and canonical in domain_config.get("attributes", {}):
            polarity = float(attr.get("polarity", 1.0))
            polarity = max(-1.0, min(1.0, polarity))
            weight = float(attr.get("weight", 1.0))
            weight = max(0.5, min(2.0, weight))
            result["attributes"].append({
                "attribute": canonical,
                "polarity": polarity,
                "weight": weight,
            })

    for cat in data.get("categories", []):
        if cat in known_categories:
            result["categories"].append(cat)

    for constraint in data.get("constraints", []):
        if all(k in constraint for k in ("field", "op", "value")):
            result["constraints"].append(constraint)

    # --- Merge synonym expansions ---
    synonym_attrs = expand_synonyms(query_text, domain)
    existing_attr_names = {a["attribute"] for a in result["attributes"]}
    for sa in synonym_attrs:
        if sa["attribute"] in existing_attr_names:
            # Boost weight of already-parsed attributes that synonyms also match
            for a in result["attributes"]:
                if a["attribute"] == sa["attribute"]:
                    a["weight"] = min(2.0, a["weight"] + 0.2)
                    break
        else:
            # Only add synonym-expanded attributes that are in the known set
            if sa["attribute"] in domain_config.get("attributes", {}):
                result["attributes"].append(sa)

    return result


def _fallback_query_parse(query_text: str, domain_config: dict) -> dict:
    """Simple keyword-based query parsing as fallback when LLM fails."""
    query_lower = query_text.lower()
    result = {"attributes": [], "categories": [], "constraints": []}

    attributes = domain_config.get("attributes", {})
    for attr_name, attr_info in attributes.items():
        # Check positive terms
        for term in attr_info.get("positive_terms", []):
            if term.lower() in query_lower:
                result["attributes"].append({
                    "attribute": attr_name,
                    "polarity": 1.0,
                    "weight": 1.0,
                })
                break
        else:
            # Check negative terms
            for term in attr_info.get("negative_terms", []):
                if term.lower() in query_lower:
                    result["attributes"].append({
                        "attribute": attr_name,
                        "polarity": -1.0,
                        "weight": 1.0,
                    })
                    break
            else:
                # Check synonyms
                for syn in attr_info.get("synonyms", []):
                    if syn.lower() in query_lower:
                        result["attributes"].append({
                            "attribute": attr_name,
                            "polarity": 1.0,
                            "weight": 1.0,
                        })
                        break

    # Check categories
    category_synonyms = domain_config.get("category_synonyms", {})
    for cat, syns in category_synonyms.items():
        for syn in syns:
            if syn.lower() in query_lower:
                result["categories"].append(cat)
                break

    # Check for length constraints
    length_match = re.search(r'(\d{3})\s*cm\s*\+', query_lower)
    if length_match:
        result["constraints"].append({
            "field": "lengths_cm",
            "op": ">=",
            "value": int(length_match.group(1)),
        })

    return result
