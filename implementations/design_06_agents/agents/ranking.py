"""Ranking Agent -- scores and orders candidates against the parsed query.

Uses heuristic scoring based on attribute matching, sentiment, and spec
alignment. No LLM call in the default mode (Phase 1 only).
"""

from __future__ import annotations

import logging
from typing import Any

from ..state import ParsedQuery, ProductAttributes, ScoredCandidate

logger = logging.getLogger(__name__)


def ranking_agent(state: dict, top_k: int = 10) -> dict:
    """Score and rank candidate products against the parsed query.

    Args:
        state: Pipeline state with "candidates" and "parsed_query".
        top_k: Max number of results to return.

    Returns:
        Updated state with "ranked" set to sorted ScoredCandidate list.
    """
    candidates: list[ProductAttributes] = state.get("candidates", [])
    parsed: ParsedQuery | None = state.get("parsed_query")

    if not candidates:
        return {**state, "ranked": []}

    if parsed is None:
        # No parsed query -- score by avg rating
        scored = [
            ScoredCandidate(
                product=p,
                score=p.avg_rating / 5.0,
                breakdown={"avg_rating": p.avg_rating / 5.0},
            )
            for p in candidates
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return {**state, "ranked": scored[:top_k]}

    scored = []
    for product in candidates:
        score, breakdown = _score_candidate(product, parsed)
        scored.append(
            ScoredCandidate(product=product, score=score, breakdown=breakdown)
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return {**state, "ranked": scored[:top_k]}


def _score_candidate(
    product: ProductAttributes, query: ParsedQuery
) -> tuple[float, dict[str, float]]:
    """Score a single candidate against the parsed query.

    Returns (overall_score, per_attribute_breakdown).
    """
    breakdown: dict[str, float] = {}
    total_weight = 0.0
    weighted_score = 0.0

    # Score soft preferences (main scoring signal)
    for attr, desired in query.soft_preferences.items():
        weight = abs(desired) / 10.0  # Normalize weight 0-1
        actual = _get_attribute_value(product, attr)

        if actual is not None:
            match = _compute_match(actual, desired)

            # Apply sentiment boost
            sentiment = product.sentiment_scores.get(attr, 0.0)
            sentiment_boost = 1.0 + 0.2 * max(0, sentiment)

            # Apply review confidence
            confidence = min(product.review_count / 5.0, 1.0)

            attr_score = match * sentiment_boost * confidence
            attr_score = min(attr_score, 1.0)  # Cap at 1.0

            breakdown[attr] = round(attr_score, 3)
            weighted_score += attr_score * weight
            total_weight += weight

    # Penalize negative preferences
    for attr, avoidance in query.negative_preferences.items():
        weight = avoidance / 10.0
        actual = _get_attribute_value(product, attr)

        if actual is not None:
            # Higher actual value = worse match for negative pref
            if isinstance(actual, (int, float)):
                penalty = actual / 10.0  # 0-1 where 1 is worst
            else:
                penalty = 0.5  # Default penalty for categorical match
            breakdown[f"neg_{attr}"] = round(-penalty, 3)
            weighted_score -= penalty * weight
            total_weight += weight

    # Spec alignment bonus (for hard filters that passed)
    spec_bonus = _spec_alignment_score(product, query)
    if spec_bonus > 0:
        breakdown["spec_match"] = round(spec_bonus, 3)
        weighted_score += spec_bonus * 0.3
        total_weight += 0.3

    # Review quality signal
    if product.review_count > 0:
        rating_score = product.avg_rating / 5.0
        breakdown["review_quality"] = round(rating_score, 3)
        weighted_score += rating_score * 0.1
        total_weight += 0.1

    # Normalize
    if total_weight > 0:
        final_score = weighted_score / total_weight
    else:
        final_score = 0.5  # No signal, neutral score

    final_score = max(0.0, min(1.0, final_score))

    return round(final_score, 4), breakdown


def _get_attribute_value(
    product: ProductAttributes, attr: str
) -> float | str | list | None:
    """Get an attribute value from product attributes or specs."""
    # Check product attributes first
    if attr in product.attributes:
        return product.attributes[attr]

    # Check specs
    if attr in product.specs:
        return product.specs[attr]

    # Check category-based mapping
    return None


def _compute_match(actual: Any, desired: float) -> float:
    """Compute match score between actual value and desired preference.

    Both are expected on a 1-10 scale for numeric attributes.
    Preferences are treated as "the user wants this much or more" --
    exceeding the desired value is not penalized.
    Returns 0-1 score.
    """
    if isinstance(actual, (int, float)):
        actual_f = float(actual)
        if actual_f >= desired:
            # Meeting or exceeding desire is a perfect or near-perfect match
            return 1.0
        else:
            # Falling short of desire is penalized proportionally
            shortfall = desired - actual_f
            max_shortfall = desired - 1.0  # Worst case: attribute is 1
            if max_shortfall <= 0:
                return 1.0
            return max(0.0, 1.0 - (shortfall / max_shortfall))
    elif isinstance(actual, list):
        # Check if desired value name matches any list element
        # This handles terrain lists
        return 0.5  # Partial match for list presence
    elif isinstance(actual, str):
        return 0.5  # Partial match
    return 0.0


def _spec_alignment_score(
    product: ProductAttributes, query: ParsedQuery
) -> float:
    """Score how well product specs align with hard filters."""
    if not query.hard_filters:
        return 0.0

    matched = 0
    total = len(query.hard_filters)

    for key, constraint in query.hard_filters.items():
        spec_val = product.specs.get(key)
        attr_val = product.attributes.get(key)
        val = spec_val if spec_val is not None else attr_val

        if val is None:
            continue

        if isinstance(constraint, dict):
            # Range filter
            if isinstance(val, list):
                # Check if any element in list matches range
                if any(_in_range(v, constraint) for v in val):
                    matched += 1
            elif _in_range(val, constraint):
                matched += 1
        elif isinstance(val, list):
            if constraint in val:
                matched += 1
        elif val == constraint:
            matched += 1

    return matched / max(total, 1)


def _in_range(value: Any, constraint: dict) -> bool:
    """Check if a value is within a range constraint."""
    try:
        v = float(value)
        if "min" in constraint and v < float(constraint["min"]):
            return False
        if "max" in constraint and v > float(constraint["max"]):
            return False
        return True
    except (TypeError, ValueError):
        return False
