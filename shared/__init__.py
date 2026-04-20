"""Shared modules for all recommendation system implementations."""

from .interface import RecommendationResult, Recommender
from .llm_provider import LLMProvider, get_provider

__all__ = [
    "RecommendationResult",
    "Recommender",
    "LLMProvider",
    "get_provider",
]
