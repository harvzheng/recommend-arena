"""Multi-Agent Recommender -- wraps the agent pipeline behind the shared interface.

Query pipeline: understand -> retrieve -> rank -> explain
All agents are plain Python functions passing state dicts between them.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from collections import defaultdict

# Ensure shared module is importable
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .agents.extractor import extract_agent
from .agents.query_understanding import query_understanding_agent
from .agents.retrieval import retrieval_agent
from .agents.ranking import ranking_agent
from .agents.explanation import explanation_agent, build_explanation
from .store import Store

logger = logging.getLogger(__name__)


class MultiAgentRecommender:
    """Multi-agent recommendation system using plain function pipeline.

    Implements the shared Recommender protocol. Each agent is a pure function
    that receives a state dict and returns an updated state dict.
    """

    def __init__(self, explain: bool = False, storage_dir: str | None = None):
        self.explain = explain
        self._storage_dir = storage_dir or tempfile.mkdtemp(prefix="design06_")

        # Initialize LLM provider (may fail if provider unavailable)
        try:
            self._llm = get_provider()
        except Exception as e:
            logger.warning("LLM provider unavailable, using fallback mode: %s", e)
            self._llm = None

        self._store = Store(
            storage_dir=self._storage_dir, llm_provider=self._llm
        )

    def ingest(
        self, products: list[dict], reviews: list[dict], domain: str
    ) -> None:
        """Ingest product and review data for a given domain.

        Runs the Extractor Agent over grouped reviews, then persists
        structured data to SQLite and embeddings to ChromaDB.
        """
        # Clear previous data for this domain (supports re-ingestion)
        self._store.clear_domain(domain)

        # Group reviews by product_id
        reviews_by_product: dict[str, list[dict]] = defaultdict(list)
        for review in reviews:
            pid = review.get("product_id", "")
            if pid:
                reviews_by_product[pid].append(review)

        # Build product lookup
        product_lookup = {}
        for p in products:
            pid = p.get("id", p.get("product_id", ""))
            product_lookup[pid] = p

        # Run Extractor Agent for each product
        for product_data in products:
            pid = product_data.get("id", product_data.get("product_id", ""))
            product_reviews = reviews_by_product.get(pid, [])

            product_attrs = extract_agent(
                product_data=product_data,
                reviews=product_reviews,
                domain=domain,
                llm_provider=self._llm,
            )

            self._store.store_product(product_attrs, product_reviews)

        logger.info(
            "Ingested %d products and %d reviews for domain '%s'",
            len(products),
            len(reviews),
            domain,
        )

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        """Run the query pipeline and return normalized results.

        Pipeline: understand -> retrieve -> rank -> explain
        """
        # Initialize pipeline state
        state: dict = {
            "raw_query": query_text,
            "domain": domain,
            "parsed_query": None,
            "candidates": [],
            "ranked": [],
            "errors": [],
        }

        # Step 1: Query Understanding (LLM-powered with fallback)
        state = query_understanding_agent(state, llm_provider=self._llm)

        # Step 2: Retrieval (deterministic)
        state = retrieval_agent(state, store=self._store)

        # Step 3: Ranking (heuristic scoring)
        state = ranking_agent(state, top_k=top_k)

        # Step 4: Explanation (lightweight, no LLM)
        state = explanation_agent(state)

        # Log any errors from the pipeline
        if state.get("errors"):
            for err in state["errors"]:
                logger.warning("Pipeline error: %s", err)

        # Convert internal ScoredCandidates to the common interface
        results = []
        for candidate in state.get("ranked", [])[:top_k]:
            # Build matched_attributes: only positive-scoring attributes
            matched = {
                k: v
                for k, v in candidate.breakdown.items()
                if not k.startswith("neg_") and v > 0
            }

            results.append(
                RecommendationResult(
                    product_id=candidate.product.product_id,
                    product_name=candidate.product.product_name,
                    score=candidate.score,
                    explanation=candidate.explanation,
                    matched_attributes=matched,
                )
            )

        return results
