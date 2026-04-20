"""Design #12: Distilled LLM Ranker Recommendation System."""

from pathlib import Path

from .recommender import DistilledLLMRecommender

_IMPL_DIR = Path(__file__).parent


def create_recommender() -> DistilledLLMRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return DistilledLLMRecommender(
        adapter_path=str(_IMPL_DIR / "trained_adapter"),
        db_path=str(_IMPL_DIR / "benchmark.db"),
    )
