"""Individual signal scoring utilities.

Handles query parsing and per-signal score computation for the ensemble.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsed query representation
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    """Result of parsing a natural language query into structured components."""

    free_text: str = ""
    soft_preferences: list[tuple[str, float]] = field(default_factory=list)
    negations: list[str] = field(default_factory=list)
    hard_constraints: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Known aspects / attributes for matching
# ---------------------------------------------------------------------------

KNOWN_ASPECTS = {
    "stiffness", "stiff", "soft", "flex",
    "damp", "dampness", "damping", "smooth",
    "edge_grip", "edge grip", "grip", "edge hold", "ice",
    "stability", "stable", "stability_at_speed",
    "playfulness", "playful", "fun", "lively",
    "powder_float", "powder", "float", "deep snow",
    "forgiveness", "forgiving", "easy", "beginner",
    "lightweight", "light", "weight", "heavy",
    "versatile", "versatility", "all-mountain", "all mountain",
    "carving", "carve", "on-piste", "on piste", "piste",
    "freeride", "off-piste", "off piste", "backcountry",
    "touring", "uphill",
    "park", "freestyle", "tricks",
    "responsive", "precise", "precision",
    "cushion", "cushioning", "comfortable",
}

# Map various query terms to canonical aspect names
ASPECT_CANONICALIZE = {
    "stiff": "stiffness",
    "soft": "stiffness",
    "flex": "stiffness",
    "damp": "dampness",
    "damping": "dampness",
    "smooth": "dampness",
    "edge grip": "edge_grip",
    "grip": "edge_grip",
    "edge hold": "edge_grip",
    "ice": "edge_grip",
    "stable": "stability",
    "stability_at_speed": "stability",
    "playful": "playfulness",
    "fun": "playfulness",
    "lively": "playfulness",
    "powder": "powder_float",
    "float": "powder_float",
    "deep snow": "powder_float",
    "forgiving": "forgiveness",
    "easy": "forgiveness",
    "beginner": "forgiveness",
    "lightweight": "weight",
    "light": "weight",
    "heavy": "weight",
    "versatile": "versatility",
    "all-mountain": "versatility",
    "all mountain": "versatility",
    "carving": "carving",
    "carve": "carving",
    "on-piste": "carving",
    "on piste": "carving",
    "piste": "carving",
    "freeride": "freeride",
    "off-piste": "freeride",
    "off piste": "freeride",
    "backcountry": "freeride",
    "touring": "touring",
    "uphill": "touring",
    "park": "freestyle",
    "freestyle": "freestyle",
    "tricks": "freestyle",
    "responsive": "responsiveness",
    "precise": "precision",
    "precision": "precision",
    "cushion": "cushioning",
    "cushioning": "cushioning",
    "comfortable": "cushioning",
}

# Aspects where the query term implies negative sentiment
NEGATIVE_POLARITY_TERMS = {"soft", "heavy"}


def parse_query(query_text: str) -> ParsedQuery:
    """Parse a natural language query into structured components.

    Extracts:
    - soft_preferences: (canonical_aspect, desired_polarity) pairs
    - negations: aspects the user does NOT want
    - hard_constraints: structured constraints (length, category, etc.)
    - free_text: the raw query for BM25/vector scoring
    """
    parsed = ParsedQuery(free_text=query_text)
    lower = query_text.lower()

    # Extract negations (e.g., "not stiff", "no park", "avoid ice")
    neg_patterns = [
        r"\bnot\s+(\w+)",
        r"\bno\s+(\w+)",
        r"\bavoid\s+(\w+)",
        r"\bwithout\s+(\w+)",
        r"\bnon[- ](\w+)",
    ]
    negated_terms: set[str] = set()
    for pattern in neg_patterns:
        for match in re.finditer(pattern, lower):
            term = match.group(1)
            negated_terms.add(term)
            canonical = ASPECT_CANONICALIZE.get(term)
            if canonical:
                # "not soft" means user WANTS stiffness (opposite), not
                # "avoid stiffness". Only add to negations if the negated
                # term has positive polarity (e.g., "not playful" = avoid
                # playfulness). For negative-polarity terms (soft, heavy),
                # negating them means the user wants the positive version.
                if term in NEGATIVE_POLARITY_TERMS:
                    # "not soft" -> wants stiffness (add as positive pref)
                    if not any(c == canonical for c, _ in parsed.soft_preferences):
                        parsed.soft_preferences.append((canonical, 1.0))
                else:
                    parsed.negations.append(canonical)

    # Extract hard constraints (length, width, etc.)
    length_match = re.search(r"(\d{3})\s*cm", lower)
    if length_match:
        parsed.hard_constraints["length_cm"] = length_match.group(1)

    length_range = re.search(r"(\d{3})\s*\+", lower)
    if length_range:
        parsed.hard_constraints["min_length_cm"] = length_range.group(1)

    # Extract soft preferences from known aspect terms
    for term, canonical in ASPECT_CANONICALIZE.items():
        if term in lower and term not in negated_terms and canonical not in parsed.negations:
            polarity = -1.0 if term in NEGATIVE_POLARITY_TERMS else 1.0
            # Avoid duplicate canonical aspects
            if not any(c == canonical for c, _ in parsed.soft_preferences):
                parsed.soft_preferences.append((canonical, polarity))

    # Extract category hints
    category_terms = {
        "race": "race",
        "slalom": "race_slalom",
        "gs": "race_gs",
        "giant slalom": "race_gs",
        "all-mountain": "all_mountain",
        "all mountain": "all_mountain",
        "freeride": "freeride",
        "freestyle": "freestyle",
        "park": "freestyle",
        "touring": "touring",
        "backcountry": "backcountry",
        "beginner": "beginner",
        "carving": "carving",
        "powder": "powder",
    }
    for term, cat in category_terms.items():
        if term in lower:
            parsed.hard_constraints.setdefault("category_hint", cat)
            break

    return parsed
