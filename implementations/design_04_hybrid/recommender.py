"""Main Recommender implementation for Design 04: Hybrid Structured + Vector.

Implements the shared Recommender protocol with dual-track architecture:
  Track 1 (Structured): ABSA extraction -> ontology normalization -> SQLite
  Track 2 (Semantic): Product summaries -> embedding -> ChromaDB
  Convergence: Vector recall -> structured filter -> multi-signal ranking
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .extractor import extract_aspects
from .normalizer import Normalizer
from .ranker import (
    ParsedQuery,
    build_explanation,
    parse_query,
    score_product,
)
from .store import StructuredStore
from .vectors import VectorIndex

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Hybrid Structured + Vector recommendation system.

    Satisfies the shared Recommender protocol.
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        db_path: str = ":memory:",
        vector_persist_dir: str | None = None,
    ):
        self.llm = llm or get_provider()
        self.store = StructuredStore(db_path=db_path)
        self.vector_index = VectorIndex(
            llm=self.llm, persist_dir=vector_persist_dir
        )
        self._normalizers: dict[str, Normalizer] = {}
        self._ingested_domains: set[str] = set()

    def _get_normalizer(self, domain: str) -> Normalizer:
        if domain not in self._normalizers:
            self._normalizers[domain] = Normalizer(domain=domain, llm=self.llm)
        return self._normalizers[domain]

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self, products: list[dict], reviews: list[dict], domain: str
    ) -> None:
        """Ingest product metadata and reviews into both stores.

        Args:
            products: Product dicts with keys like product_id/id, product_name/name, etc.
            reviews: Review dicts with keys like product_id, review_text/text, etc.
            domain: Domain identifier for ontology lookup.
        """
        normalizer = self._get_normalizer(domain)

        # Clear previous data for this domain to support re-ingestion
        if domain in self._ingested_domains:
            self.store.clear_domain(domain)
            self.vector_index.clear_domain(domain)
        self._ingested_domains.add(domain)

        # Step 1: Ingest products
        product_names: dict[str, str] = {}
        for p in products:
            pid = p.get("product_id") or p.get("id", "")
            pname = p.get("product_name") or p.get("name", "")
            metadata = {}
            # Gather all metadata
            for key in ("specs", "attributes", "metadata", "brand", "category"):
                if key in p:
                    metadata[key] = p[key]
            self.store.upsert_product(pid, pname, domain, metadata)
            product_names[pid] = pname

        # Step 2: Ingest reviews with ABSA extraction
        for review in reviews:
            pid = review.get("product_id", "")
            text = review.get("review_text") or review.get("text", "")
            author = review.get("author") or review.get("reviewer")
            source = review.get("source", "")
            if not text:
                continue

            review_id = self.store.insert_review(
                product_id=pid,
                text=text,
                author=author,
                source=source,
            )

            # ABSA extraction
            raw_aspects = extract_aspects(text, domain, self.llm)

            # Normalize and store aspects
            normalized_aspects = []
            for a in raw_aspects:
                norm = normalizer.normalize(a["aspect"])
                normalized_aspects.append(
                    {
                        "raw_aspect": a["aspect"],
                        "norm_aspect": norm,
                        "opinion": a["opinion"],
                        "sentiment": a["sentiment"],
                    }
                )

            if normalized_aspects:
                self.store.insert_aspects(review_id, pid, normalized_aspects)

        # Step 3: Build and store embeddings for all products
        for pid, pname in product_names.items():
            summary = self._build_summary_text(pid, pname, domain)
            self.vector_index.upsert_product(pid, domain, summary, pname)

        logger.info(
            "Ingested %d products and %d reviews for domain %r",
            len(products),
            len(reviews),
            domain,
        )

    def _build_summary_text(
        self, product_id: str, product_name: str, domain: str
    ) -> str:
        """Build a summary text for embedding from aggregated aspect data."""
        rows = self.store.get_product_summary_data(product_id)
        if not rows:
            return f"{product_name}: {domain} product"

        parts: list[str] = [f"{product_name}:"]
        for norm_aspect, avg_sentiment, mention_count in rows:
            if norm_aspect.startswith("_unmatched:"):
                aspect_label = norm_aspect.split(":", 1)[1]
            elif norm_aspect.startswith("terrain:"):
                aspect_label = norm_aspect.split(":", 1)[1].replace("_", " ")
                parts.append(aspect_label)
                continue
            else:
                aspect_label = norm_aspect.replace("_", " ")

            # Convert sentiment to descriptive word
            if avg_sentiment > 0.6:
                desc = "excellent"
            elif avg_sentiment > 0.3:
                desc = "good"
            elif avg_sentiment > -0.3:
                desc = "moderate"
            elif avg_sentiment > -0.6:
                desc = "weak"
            else:
                desc = "poor"

            parts.append(f"{desc} {aspect_label}")

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        domain: str,
        top_k: int = 10,
    ) -> list[RecommendationResult]:
        """Query for product recommendations using hybrid retrieval.

        Args:
            query_text: Natural language query.
            domain: Domain to search in.
            top_k: Number of results to return.

        Returns:
            List of RecommendationResult sorted by score descending.
        """
        normalizer = self._get_normalizer(domain)
        weights = normalizer.get_weights()

        # Step 1: Parse query
        parsed = parse_query(query_text, domain, normalizer)
        logger.debug(
            "Parsed query: aspects=%s, negations=%s, numeric=%s",
            parsed.required_aspects,
            parsed.negations,
            parsed.numeric_filters,
        )

        # Step 2: Vector recall (broad candidate set)
        recall_k = min(max(top_k * 5, 50), 100)
        candidate_ids, similarities = self.vector_index.query(
            query_text=parsed.semantic_text,
            domain=domain,
            top_k=recall_k,
        )

        if not candidate_ids:
            return []

        # Build similarity lookup
        sim_map = dict(zip(candidate_ids, similarities))

        # Step 3: Structured filter (apply negations)
        if parsed.negations:
            filtered_ids = self.store.filter_by_negations(
                candidate_ids, parsed.negations
            )
        else:
            filtered_ids = candidate_ids

        if not filtered_ids:
            # If all candidates were filtered, fall back to original set
            # (negation filter might be too aggressive)
            filtered_ids = candidate_ids

        # Step 4: Score and rank
        scored: list[tuple[str, float, dict[str, float]]] = []
        for pid in filtered_ids:
            vsim = sim_map.get(pid, 0.0)
            score, breakdown = score_product(pid, parsed, vsim, self.store, weights)
            scored.append((pid, score, breakdown))

        # Sort by score descending
        scored.sort(key=lambda x: -x[1])

        # Step 5: Build results
        results: list[RecommendationResult] = []
        for pid, score, breakdown in scored[:top_k]:
            pname = self.store.get_product_name(pid)
            explanation = build_explanation(pname, breakdown, parsed, weights)

            # Extract matched_attributes for the interface
            matched_attrs: dict[str, float] = {}
            for key, value in breakdown.items():
                if key.startswith("aspect:"):
                    attr_name = key.split(":", 1)[1]
                    matched_attrs[attr_name] = value

            results.append(
                RecommendationResult(
                    product_id=pid,
                    product_name=pname,
                    score=score,
                    explanation=explanation,
                    matched_attributes=matched_attrs,
                )
            )

        return results
