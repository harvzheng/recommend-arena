"""Design #9: Faceted Search + SQLite FTS5 Recommendation System."""

from .recommender import FacetedSearchRecommender


def create_recommender() -> FacetedSearchRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return FacetedSearchRecommender()
