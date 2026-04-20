"""Explanation Agent -- builds human-readable explanations from score breakdowns.

This agent is lightweight and does NOT use LLM calls. It constructs explanations
directly from the ranking agent's per-attribute score breakdown.
"""

from __future__ import annotations

from ..state import ScoredCandidate


def explanation_agent(state: dict) -> dict:
    """Add explanations to ranked candidates from their score breakdowns.

    Args:
        state: Pipeline state with "ranked" list of ScoredCandidate.

    Returns:
        Updated state with explanations filled in on each candidate.
    """
    ranked: list[ScoredCandidate] = state.get("ranked", [])

    for candidate in ranked:
        if not candidate.explanation:
            candidate.explanation = build_explanation(candidate)

    return state


def build_explanation(candidate: ScoredCandidate) -> str:
    """Build a human-readable explanation from a score breakdown.

    Args:
        candidate: Scored candidate with breakdown dict.

    Returns:
        Human-readable explanation string.
    """
    breakdown = candidate.breakdown
    product = candidate.product

    if not breakdown:
        return f"General match based on {product.product_name} profile."

    # Separate positive matches, negative penalties, and meta scores
    positives = []
    negatives = []
    meta = []

    for attr, score in sorted(breakdown.items(), key=lambda x: -abs(x[1])):
        if attr.startswith("neg_"):
            real_attr = attr[4:]
            negatives.append((real_attr, abs(score)))
        elif attr in ("spec_match", "review_quality"):
            meta.append((attr, score))
        else:
            positives.append((attr, score))

    parts = []

    # Top positive matches
    strong = [(a, s) for a, s in positives if s >= 0.7]
    moderate = [(a, s) for a, s in positives if 0.4 <= s < 0.7]
    weak = [(a, s) for a, s in positives if s < 0.4]

    if strong:
        attrs = ", ".join(
            f"{_format_attr(a)} ({s:.0%})" for a, s in strong[:4]
        )
        parts.append(f"Strong match on {attrs}")

    if moderate:
        attrs = ", ".join(_format_attr(a) for a, s in moderate[:3])
        parts.append(f"Moderate fit for {attrs}")

    if negatives:
        neg_attrs = ", ".join(
            f"{_format_attr(a)} ({s:.0%} penalty)" for a, s in negatives[:2]
        )
        parts.append(f"Potential concern: {neg_attrs}")

    for attr, score in meta:
        if attr == "spec_match" and score >= 0.8:
            parts.append("Specs match your requirements")
        elif attr == "review_quality" and score >= 0.8:
            parts.append("Highly rated by reviewers")

    if parts:
        return ". ".join(parts) + "."
    else:
        return f"Matched based on overall profile similarity."


def _format_attr(attr: str) -> str:
    """Format an attribute name for human display."""
    return attr.replace("_", " ").replace("at speed", "at speed")
