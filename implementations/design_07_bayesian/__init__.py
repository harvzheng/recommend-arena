"""Design #7: Bayesian / Probabilistic Scoring Recommendation System."""

from .recommender import BayesianRecommender


def create_recommender() -> BayesianRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return BayesianRecommender()
