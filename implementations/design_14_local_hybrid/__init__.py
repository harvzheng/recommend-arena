"""Design #14 — Local-First Hybrid (architecture spike, off-the-shelf).

Pipeline: filter parser → hard prefilter → parallel FTS5 + vector retrieval
→ Reciprocal Rank Fusion → optional cross-encoder rerank → return.

Hot path is Rust (`arena_core`); ML model invocations stay in Python.
"""

from .recommender import LocalHybridRecommender


def create_recommender() -> LocalHybridRecommender:
    """Factory function expected by the benchmark runner."""
    return LocalHybridRecommender()


__all__ = ["LocalHybridRecommender", "create_recommender"]
