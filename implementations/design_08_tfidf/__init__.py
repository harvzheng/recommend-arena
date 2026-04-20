"""Design #8: TF-IDF + Learned Attribute Weights Recommendation System."""

from .recommender import TFIDFRecommender


def create_recommender() -> TFIDFRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return TFIDFRecommender()
