"""Design #11: Fine-Tuned Embedding Ranker."""

from pathlib import Path

from .recommender import FineTunedEmbeddingRecommender

_IMPL_DIR = Path(__file__).parent


def create_recommender() -> FineTunedEmbeddingRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return FineTunedEmbeddingRecommender(
        model_dir=str(_IMPL_DIR / "trained_model"),
    )
