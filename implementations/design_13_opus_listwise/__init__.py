"""Design #13 — Frontier Listwise (Opus 4.7).

Sends the entire product catalog (with reviews) into one prompt with prompt
caching, asks Opus 4.7 to listwise-rank, returns the top-k IDs. The arena's
upper-bound reference, not a production design.
"""

from .recommender import OpusListwiseRecommender


def create_recommender() -> OpusListwiseRecommender:
    """Factory function expected by the benchmark runner."""
    return OpusListwiseRecommender()


__all__ = ["OpusListwiseRecommender", "create_recommender"]
