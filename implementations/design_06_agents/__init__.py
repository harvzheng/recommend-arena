"""Design #6: Multi-Agent Architecture for product recommendations.

A recommendation system built around specialized, collaborating agents:
- Extractor Agent: parses reviews into structured attributes
- Query Understanding Agent: normalizes user intent into structured query
- Retrieval Agent: fetches candidates via vector similarity + hard filters
- Ranking Agent: scores candidates against the parsed query
- Explanation Agent: builds human-readable explanations (no LLM)

All agents are plain Python functions passing state dicts between them.
No LangGraph dependency.
"""

from .recommender import MultiAgentRecommender


def create_recommender() -> MultiAgentRecommender:
    """Factory function for benchmark runner discovery."""
    return MultiAgentRecommender()
