"""Design #10: Ensemble / Learning to Rank Recommendation System."""

from .recommender import EnsembleLTRRecommender


def create_recommender() -> EnsembleLTRRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return EnsembleLTRRecommender()
