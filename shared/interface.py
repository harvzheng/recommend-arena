"""Shared interface for all recommendation system implementations.

All 10 designs must implement the Recommender protocol.
The benchmark runner uses this interface to evaluate each design uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class RecommendationResult:
    """A single recommendation result returned by a recommender."""

    product_id: str
    product_name: str
    score: float  # 0.0 to 1.0, normalized
    explanation: str  # Human-readable explanation of why this product was recommended
    matched_attributes: dict[str, float] = field(
        default_factory=dict
    )  # attribute -> match strength (0-1)


@runtime_checkable
class Recommender(Protocol):
    """Protocol that all recommendation system implementations must satisfy."""

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Ingest product and review data for a given domain.

        Args:
            products: List of product dicts with at minimum:
                - product_id: str
                - product_name: str
                - domain: str
                - metadata: dict (domain-specific specs like length_cm, weight_g, etc.)
            reviews: List of review dicts with at minimum:
                - review_id: str
                - product_id: str
                - review_text: str
                - source: str (optional)
            domain: str - the product domain ("ski", "running_shoe", etc.)
        """
        ...

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        """Query the recommendation system with natural language.

        Args:
            query_text: Natural language query (e.g., "stiff, on-piste carving ski, 180cm+")
            domain: Product domain to search in
            top_k: Number of results to return

        Returns:
            List of RecommendationResult, sorted by score descending
        """
        ...
