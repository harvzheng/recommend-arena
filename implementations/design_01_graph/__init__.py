"""Design #1: Graph-Based Knowledge Graph Recommender.

Exports a create_recommender() factory function for the benchmark runner.
"""

from .recommender import GraphRecommender


def create_recommender() -> GraphRecommender:
    """Factory function used by the benchmark runner to instantiate this recommender."""
    return GraphRecommender()


__all__ = ["GraphRecommender", "create_recommender"]
