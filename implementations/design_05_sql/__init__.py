"""Design #5: SQL-First / SQLite + FTS5 Recommendation System."""

from .recommender import SqlRecommender


def create_recommender() -> SqlRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return SqlRecommender()
