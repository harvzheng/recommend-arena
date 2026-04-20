"""Design #11: Fine-Tuned Embedding Ranker.

The embedding space IS the recommendation engine. Query-time cost is one
encoder forward pass + cosine similarity. No LLM calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult

from .pairs import (
    ANCHOR_TEMPLATES,
    ProductProfile,
    TrainingConfig,
    generate_contrastive_pairs,
)
from .ranker import explain_match, rank_products
from .trainer import export_model, fine_tune
from .vectors import (
    AttributeCentroid,
    ProductVector,
    build_attribute_centroids,
    build_product_vectors,
    chunk_review,
)

logger = logging.getLogger(__name__)


class FineTunedEmbeddingRecommender:
    """Design #11: Fine-tuned embedding ranker."""

    def __init__(self, model_dir: str | None = None):
        self.model = None
        self.model_dir = model_dir
        self.product_vectors: dict[str, list[ProductVector]] = {}
        self.centroids: dict[str, list[AttributeCentroid]] = {}
        self._trained = False

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        from sentence_transformers import SentenceTransformer

        profiles = self._build_profiles(products, reviews, domain)

        if self.model is None:
            if self.model_dir and Path(self.model_dir, "model").exists():
                logger.info("Loading pre-trained model from %s", self.model_dir)
                self.model = SentenceTransformer(str(Path(self.model_dir) / "model"))
                self._trained = True
            else:
                logger.info("Training new model from %d profiles", len(profiles))
                self.model = self._train(profiles)
                self._trained = True

        pvecs = build_product_vectors(products, reviews, self.model)
        self.product_vectors[domain] = pvecs

        attributes = list(ANCHOR_TEMPLATES.keys())
        self.centroids[domain] = build_attribute_centroids(pvecs, attributes)
        logger.info(
            "Ingested %d products for domain '%s' (%d centroids)",
            len(pvecs), domain, len(self.centroids[domain]),
        )

    def query(
        self, query_text: str, domain: str, top_k: int = 10,
    ) -> list[RecommendationResult]:
        if domain not in self.product_vectors or self.model is None:
            return []

        pvecs = self.product_vectors[domain]
        centroids = self.centroids.get(domain, [])

        query_vec = self.model.encode(query_text, normalize_embeddings=True)

        ranked = rank_products(query_text, pvecs, self.model, top_k=top_k)

        results = []
        for pv, score in ranked:
            explanation, matched_attrs = explain_match(
                query_vec, pv.vector, centroids,
            )
            results.append(RecommendationResult(
                product_id=pv.product_id,
                product_name=pv.product_name,
                score=round(score, 4),
                explanation=explanation,
                matched_attributes=matched_attrs,
            ))

        return results

    def _build_profiles(
        self,
        products: list[dict],
        reviews: list[dict],
        domain: str,
    ) -> list[ProductProfile]:
        reviews_by_product: dict[str, list[str]] = {}
        for r in reviews:
            pid = r.get("product_id", r.get("id", ""))
            text = r.get("review_text", r.get("text", ""))
            if pid and text:
                if pid not in reviews_by_product:
                    reviews_by_product[pid] = []
                reviews_by_product[pid].append(text)

        profiles = []
        for p in products:
            pid = p.get("product_id", p.get("id", ""))
            pname = p.get("product_name", p.get("name", ""))

            passages = []
            for text in reviews_by_product.get(pid, []):
                passages.extend(chunk_review(text))

            profiles.append(ProductProfile(
                product_id=pid,
                product_name=pname,
                domain=domain,
                attribute_scores=p.get("attributes", {}),
                review_passages=passages,
                metadata=p.get("specs", {}),
            ))
        return profiles

    def _train(self, profiles: list[ProductProfile]):
        config = TrainingConfig()
        if self.model_dir:
            config.output_dir = self.model_dir

        pairs = generate_contrastive_pairs(profiles)
        logger.info("Generated %d contrastive pairs", len(pairs))

        model = fine_tune(pairs, config)
        export_model(model, config)
        return model
