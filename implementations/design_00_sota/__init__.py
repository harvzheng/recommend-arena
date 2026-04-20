"""Design #0: SOTA Control — Retrieve-and-Rerank Recommender."""

from .recommender import SotaRecommender


def create_recommender() -> SotaRecommender:
    """Factory function used by the benchmark runner."""
    return SotaRecommender()
