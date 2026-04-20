"""Feature vector assembly for (query, product) pairs.

Combines signals from BM25, FAISS, and SQLite into a feature vector
for the XGBoost meta-ranker.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .indices import BM25Index, FAISSIndex, StructuredStore
    from .scorer import ParsedQuery

logger = logging.getLogger(__name__)

# Feature names matching the vector positions (used for explanations)
FEATURE_NAMES = [
    "BM25 relevance",
    "vector similarity",
    "attribute match rate",
    "attribute sentiment",
    "negation penalty",
    "review count",
    "avg rating",
    "attribute coverage",
    "hard filter pass",
    "sentiment gap",
]

NUM_FEATURES = len(FEATURE_NAMES)


class FeatureAssembler:
    """Assembles per-candidate feature vectors from multiple retrieval signals."""

    def __init__(
        self,
        bm25_index: BM25Index,
        faiss_index: FAISSIndex,
        store: StructuredStore,
        embed_fn,
        domain: str,
    ) -> None:
        self.bm25 = bm25_index
        self.faiss = faiss_index
        self.store = store
        self.embed_fn = embed_fn
        self.domain = domain
        self._total_aspects = max(self.store.get_total_aspects(domain), 1)

    def build_features(
        self,
        parsed_query: ParsedQuery,
        candidate_ids: list[str],
    ) -> np.ndarray:
        """Build feature matrix for all candidates.

        Returns:
            ndarray of shape (len(candidate_ids), NUM_FEATURES).
        """
        if not candidate_ids:
            return np.zeros((0, NUM_FEATURES), dtype=np.float32)

        query_text = parsed_query.free_text

        # Batch retrieval scores
        bm25_scores = self.bm25.score_all(query_text)
        query_embedding = self.embed_fn(query_text)
        vector_scores = self.faiss.score_all(query_embedding)

        rows = []
        for pid in candidate_ids:
            attr_feats = self._attribute_features(parsed_query, pid)
            prod_feats = self._product_features(pid)

            row = [
                bm25_scores.get(pid, 0.0),
                vector_scores.get(pid, 0.0),
                attr_feats["match_rate"],
                attr_feats["sentiment_avg"],
                attr_feats["negation_violation"],
                prod_feats["review_count"],
                prod_feats["avg_rating"],
                prod_feats["attribute_coverage"],
                float(attr_feats["hard_filter_pass"]),
                attr_feats["sentiment_gap"],
            ]
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    def _attribute_features(self, parsed_query: ParsedQuery, product_id: str) -> dict:
        """Score how well a product's extracted attributes match query preferences."""
        prefs = parsed_query.soft_preferences
        negs = parsed_query.negations

        scores = self.store.get_attribute_scores(product_id)
        matched = 0
        sentiments = []

        for aspect, desired_polarity in prefs:
            if aspect in scores:
                matched += 1
                sentiments.append(scores[aspect]["avg_sentiment"])

        neg_violation = 0.0
        for neg_aspect in negs:
            if neg_aspect in scores and scores[neg_aspect]["mention_count"] > 0:
                # The product has mentions of an aspect the user wants to avoid
                neg_violation = 1.0
                break

        match_rate = matched / max(len(prefs), 1)
        sentiment_avg = float(np.mean(sentiments)) if sentiments else 0.0
        sentiment_gap = abs(1.0 - sentiment_avg) if sentiments else 1.0

        return {
            "match_rate": match_rate,
            "sentiment_avg": sentiment_avg,
            "negation_violation": neg_violation,
            "hard_filter_pass": neg_violation == 0.0,
            "sentiment_gap": sentiment_gap,
        }

    def _product_features(self, product_id: str) -> dict:
        """Get product-level features from the structured store."""
        stats = self.store.get_review_stats(product_id)
        attr_scores = self.store.get_attribute_scores(product_id)
        coverage = len(attr_scores) / self._total_aspects if self._total_aspects > 0 else 0.0

        return {
            "review_count": float(stats["review_count"]),
            "avg_rating": float(stats["avg_rating"]),
            "attribute_coverage": coverage,
        }
