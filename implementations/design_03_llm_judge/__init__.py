"""Design #3 — LLM-as-Judge recommendation system.

Uses embedding recall (ChromaDB) for candidate retrieval and pointwise
LLM scoring for precise ranking.  All LLM calls go through the shared
provider abstraction.
"""

from .recommender import LLMJudgeRecommender


def create_recommender() -> LLMJudgeRecommender:
    """Factory function expected by the benchmark runner."""
    return LLMJudgeRecommender()


__all__ = ["LLMJudgeRecommender", "create_recommender"]
