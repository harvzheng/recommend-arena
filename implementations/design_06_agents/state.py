"""Shared state dataclasses for the multi-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProductAttributes:
    """Structured product representation used throughout the pipeline."""

    product_id: str
    product_name: str
    domain: str
    attributes: dict[str, float | str | list]  # e.g. {"stiffness": 9, "terrain": ["on-piste"]}
    sentiment_scores: dict[str, float] = field(default_factory=dict)
    review_count: int = 0
    avg_rating: float = 0.0
    specs: dict[str, Any] = field(default_factory=dict)
    category: str = ""
    brand: str = ""


@dataclass
class ParsedQuery:
    """Structured representation of a user query."""

    domain: str
    hard_filters: dict[str, Any] = field(default_factory=dict)
    soft_preferences: dict[str, float] = field(default_factory=dict)
    freetext_intent: str = ""
    negative_preferences: dict[str, float] = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    """A product with its score and per-attribute breakdown."""

    product: ProductAttributes
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


# The pipeline state is a plain dict with these keys:
# {
#     "raw_query": str,
#     "domain": str,
#     "parsed_query": ParsedQuery | None,
#     "candidates": list[ProductAttributes],
#     "ranked": list[ScoredCandidate],
#     "errors": list[str],
# }
