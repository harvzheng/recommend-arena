"""Design #2: Pure Embedding / Vector-First Recommendation System."""

from .recommender import EmbeddingRecommender


def create_recommender() -> EmbeddingRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return EmbeddingRecommender()
