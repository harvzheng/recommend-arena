"""Main Recommender implementation for Design #2: Pure Embedding / Vector-First.

Uses the shared LLM provider for embeddings and ChromaDB for vector storage.
Implements contrastive scoring (positive vs negative query) as the key differentiator.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import chromadb

# Ensure shared module is importable
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .embedder import build_index
from .retrieval import retrieve_and_score

logger = logging.getLogger(__name__)


class EmbeddingRecommender:
    """Pure embedding recommender with contrastive scoring.

    Implements the shared Recommender protocol:
        - ingest(products, reviews, domain) -> None
        - query(query_text, domain, top_k) -> list[RecommendationResult]
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider or get_provider()
        self._chroma = chromadb.EphemeralClient()  # in-memory, no singleton conflict
        self._product_names: dict[str, dict[str, str]] = {}  # domain -> {pid: name}

    def ingest(
        self,
        products: list[dict],
        reviews: list[dict],
        domain: str,
    ) -> None:
        """Ingest product and review data for a given domain.

        Chunks reviews into opinion-bearing passages, embeds them via the
        shared LLM provider, and upserts into a ChromaDB collection scoped
        by domain. Idempotent -- re-ingesting replaces the previous index.
        """
        logger.info(
            "Ingesting domain=%s: %d products, %d reviews",
            domain,
            len(products),
            len(reviews),
        )
        collection, product_names = build_index(
            client=self._chroma,
            provider=self._provider,
            products=products,
            reviews=reviews,
            domain=domain,
        )
        self._product_names[domain] = product_names
        logger.info(
            "Ingestion complete for domain=%s: %d passages indexed",
            domain,
            collection.count(),
        )

    def query(
        self,
        query_text: str,
        domain: str,
        top_k: int = 10,
    ) -> list[RecommendationResult]:
        """Query the recommendation system with natural language.

        Uses contrastive scoring (positive vs negative query) to rank products.
        Returns up to top_k results with scores normalized to 0-1.
        """
        product_names = self._product_names.get(domain, {})

        try:
            collection = self._chroma.get_collection(f"passages_{domain}")
        except Exception:
            logger.warning("No collection found for domain=%s. Run ingest first.", domain)
            return []

        raw_results = retrieve_and_score(
            collection=collection,
            provider=self._provider,
            query_text=query_text,
            domain=domain,
            product_names=product_names,
            top_k=top_k,
        )

        # Convert to RecommendationResult
        output: list[RecommendationResult] = []
        for r in raw_results:
            output.append(
                RecommendationResult(
                    product_id=r["product_id"],
                    product_name=r["product_name"],
                    score=r["score"],
                    explanation=r["explanation"],
                    matched_attributes=r["matched_attributes"],
                )
            )

        return output
