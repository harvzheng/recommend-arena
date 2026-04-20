"""Main Recommender implementation for Design #3 — LLM-as-Judge."""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import chromadb

# Ensure the project root is importable so ``shared`` resolves
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.interface import RecommendationResult  # noqa: E402
from shared.llm_provider import LLMProvider, get_provider  # noqa: E402

from .cache import JudgeCache  # noqa: E402
from .extractor import extract_profile  # noqa: E402
from .judge import build_matched_attributes, judge_pointwise  # noqa: E402

logger = logging.getLogger(__name__)


class LLMJudgeRecommender:
    """Implements the shared ``Recommender`` protocol using the LLM-as-Judge
    pipeline: embedding recall via ChromaDB followed by pointwise LLM scoring.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        cache_path: str | Path | None = None,
        chroma_path: str | Path | None = None,
        recall_k: int = 20,
    ) -> None:
        self.provider = provider or get_provider()
        self.recall_k = recall_k

        # Persistent cache
        if cache_path is None:
            self._tmp_cache = tempfile.NamedTemporaryFile(
                suffix=".sqlite", delete=False
            )
            cache_path = self._tmp_cache.name
        else:
            self._tmp_cache = None
        self.cache = JudgeCache(db_path=cache_path)

        # ChromaDB (persistent or ephemeral)
        if chroma_path is not None:
            self._chroma_client = chromadb.PersistentClient(
                path=str(chroma_path)
            )
        else:
            self._chroma_client = chromadb.EphemeralClient()  # in-memory, no singleton conflict

        # We create / get one collection per domain lazily
        self._collections: dict[str, Any] = {}

        # In-memory store of product metadata keyed by product_id
        self._products: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------
    def _get_collection(self, domain: str):
        if domain not in self._collections:
            self._collections[domain] = (
                self._chroma_client.get_or_create_collection(
                    name=f"products_{domain}",
                    metadata={"hnsw:space": "cosine"},
                )
            )
        return self._collections[domain]

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(
        self,
        products: list[dict],
        reviews: list[dict],
        domain: str,
    ) -> None:
        """Ingest products and reviews for *domain*.

        For each product:
        1. Run LLM extraction on its reviews to build a structured profile.
        2. Embed the profile summary into ChromaDB for recall.
        """
        collection = self._get_collection(domain)

        # Index reviews by product_id for fast lookup
        reviews_by_product: dict[str, list[str]] = {}
        for rev in reviews:
            pid = rev.get("product_id", "")
            text = rev.get("review_text") or rev.get("text", "")
            if pid and text:
                reviews_by_product.setdefault(pid, []).append(text)

        for product in products:
            pid = product.get("product_id") or product.get("id", "")
            pname = product.get("product_name") or product.get("name", "")
            if not pid:
                continue

            product_reviews = reviews_by_product.get(pid, [])

            # Extract structured profile via LLM (cached)
            extraction = extract_profile(
                product_id=pid,
                product_name=pname,
                domain=domain,
                reviews=product_reviews,
                provider=self.provider,
                cache=self.cache,
            )

            profile = extraction["profile"]
            summary = extraction["summary"]

            # Store product metadata for later retrieval
            self._products[pid] = {
                "product_id": pid,
                "name": pname,
                "domain": domain,
                "profile": profile,
                "summary": summary,
                "specs": product.get("specs", {}),
                "attributes": product.get("attributes", {}),
            }

            # Embed summary and upsert into ChromaDB
            try:
                embedding = self.provider.embed(summary)
                collection.upsert(
                    ids=[pid],
                    embeddings=[embedding],
                    metadatas=[{"domain": domain, "name": pname}],
                    documents=[summary],
                )
            except RuntimeError as exc:
                logger.warning(
                    "Embedding failed for %s: %s — skipping vector store",
                    pid,
                    exc,
                )

        logger.info(
            "Ingested %d products for domain '%s'", len(products), domain
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        domain: str,
        top_k: int = 10,
    ) -> list[RecommendationResult]:
        """Query the recommendation system with natural language.

        1. Embed the query and recall top-K candidates from ChromaDB.
        2. Score each candidate using the LLM judge (pointwise, cached).
        3. Return ranked results with explanations.
        """
        collection = self._get_collection(domain)

        # --- Stage 1: Embedding recall ---
        try:
            query_embedding = self.provider.embed(query_text)
        except RuntimeError as exc:
            logger.error("Query embedding failed: %s", exc)
            return []

        try:
            recall_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(self.recall_k, collection.count()),
                where={"domain": domain} if collection.count() > 0 else None,
            )
        except Exception as exc:
            logger.error("ChromaDB recall failed: %s", exc)
            return []

        if not recall_results["ids"] or not recall_results["ids"][0]:
            return []

        candidate_ids = recall_results["ids"][0]

        # --- Stage 2: LLM-as-Judge (pointwise) ---
        results: list[RecommendationResult] = []
        for pid in candidate_ids:
            product = self._products.get(pid)
            if product is None:
                continue

            judge_result = judge_pointwise(
                user_query=query_text,
                domain=domain,
                product_id=pid,
                product_name=product["name"],
                product_profile=product["profile"],
                provider=self.provider,
                cache=self.cache,
            )

            # Normalize score from 1-10 to 0-1
            normalized_score = judge_result["score"] / 10.0

            results.append(
                RecommendationResult(
                    product_id=pid,
                    product_name=product["name"],
                    score=normalized_score,
                    explanation=judge_result.get("reasoning", ""),
                    matched_attributes=build_matched_attributes(judge_result),
                )
            )

        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
