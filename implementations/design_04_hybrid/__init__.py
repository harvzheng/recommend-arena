"""Design 04: Hybrid Structured + Vector Recommendation System.

Dual-track retrieval combining ABSA-extracted structured attributes
with semantic vector embeddings, unified through a multi-signal ranking layer.
"""

from .recommender import HybridRecommender


def create_recommender() -> HybridRecommender:
    """Factory function for benchmark runner discovery."""
    return HybridRecommender()
