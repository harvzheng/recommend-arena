"""Main Recommender implementation for Design #10: Ensemble / Learning to Rank.

Combines BM25, FAISS vector search, and SQLite structured attributes through
an XGBoost LTR meta-ranker to produce final recommendations.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the project root is on sys.path so `shared` is importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .features import FEATURE_NAMES, NUM_FEATURES, FeatureAssembler
from .indices import BM25Index, FAISSIndex, StructuredStore
from .ranker import LTRRanker, build_explanation, normalize_scores
from .scorer import parse_query
from .synthetic import load_or_generate_judgments

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ABSA extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Extract product attributes from this review. For each attribute mentioned, provide the aspect name, sentiment (-1.0 to 1.0), and confidence (0.0 to 1.0).

Common aspects: stiffness, dampness, edge_grip, stability, playfulness, powder_float, forgiveness, weight, versatility, carving, freeride, precision, responsiveness, build_quality, comfort, durability, value

Review for product "{product_name}":
"{review_text}"

Respond with JSON only. Format:
{{"attributes": [{{"aspect": "<name>", "sentiment": <float>, "confidence": <float>}}]}}"""


class EnsembleLTRRecommender:
    """Recommender using ensemble LTR (Design #10).

    Three retrieval signals (BM25, FAISS vector, SQLite attributes) are combined
    through an XGBoost ranker trained on LLM-generated synthetic judgments.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm
        # Per-domain state
        self._domains: dict[str, _DomainState] = {}

    def _get_llm(self) -> LLMProvider:
        """Lazily initialize the LLM provider."""
        if self.llm is None:
            self.llm = get_provider()
        return self.llm

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Ingest product and review data, building all indices and training the ranker."""
        llm = self._get_llm()
        state = _DomainState()

        # Normalize product data
        normalized_products = []
        for p in products:
            normalized_products.append({
                "product_id": p.get("product_id", p.get("id", "")),
                "product_name": p.get("product_name", p.get("name", "")),
                "brand": p.get("brand", ""),
                "category": p.get("category", ""),
                "metadata": p.get("metadata", p.get("specs", {})),
                "attributes": p.get("attributes", {}),
            })

        # Group reviews by product
        reviews_by_product: dict[str, list[dict]] = {}
        for r in reviews:
            pid = r.get("product_id", "")
            reviews_by_product.setdefault(pid, []).append(r)

        # Step 1: ABSA extraction via LLM
        logger.info("Extracting attributes from %d reviews...", len(reviews))
        extracted_attributes = self._extract_attributes(
            normalized_products, reviews_by_product, llm
        )

        # Step 2: Build BM25 index (concatenated reviews per product)
        product_docs = {}
        for p in normalized_products:
            pid = p["product_id"]
            rev_texts = [
                r.get("review_text", r.get("text", ""))
                for r in reviews_by_product.get(pid, [])
            ]
            # Include product metadata in the document for better keyword matching
            meta_text = f"{p['product_name']} {p['brand']} {p['category']}"
            product_docs[pid] = meta_text + " " + " ".join(rev_texts)

        state.bm25_index = BM25Index()
        state.bm25_index.build(product_docs)

        # Step 3: Build FAISS index (embed concatenated reviews per product)
        logger.info("Building FAISS vector index...")
        product_embeddings = {}
        for pid, doc in product_docs.items():
            # Truncate long docs to avoid embedding model limits
            truncated = doc[:2000]
            try:
                embedding = llm.embed(truncated)
                product_embeddings[pid] = embedding
            except Exception as e:
                logger.warning("Failed to embed product %s: %s", pid, e)

        state.faiss_index = FAISSIndex()
        state.faiss_index.build(product_embeddings)

        # Step 4: Build SQLite structured store
        state.store = StructuredStore()
        state.store.build(normalized_products, extracted_attributes, domain)

        # Store review stats
        review_counts = {
            pid: len(revs) for pid, revs in reviews_by_product.items()
        }
        avg_ratings = {}
        for pid, revs in reviews_by_product.items():
            ratings = [
                r.get("rating", 0) for r in revs if r.get("rating") is not None
            ]
            avg_ratings[pid] = sum(ratings) / len(ratings) if ratings else 0.0

        state.store.set_review_stats(review_counts, avg_ratings)

        # Step 5: Build feature assembler
        state.assembler = FeatureAssembler(
            bm25_index=state.bm25_index,
            faiss_index=state.faiss_index,
            store=state.store,
            embed_fn=lambda text: llm.embed(text),
            domain=domain,
        )

        # Step 6: Generate synthetic training data and train ranker
        logger.info("Training LTR ranker...")
        state.ranker = LTRRanker()

        judgments = load_or_generate_judgments(
            products=normalized_products,
            extracted_attributes=extracted_attributes,
            domain=domain,
            llm=llm,
            products_per_query=min(15, len(normalized_products)),
        )

        if judgments:
            self._train_ranker(state, judgments, normalized_products, domain)

        # Store products for reference
        state.products = {p["product_id"]: p for p in normalized_products}
        state.domain = domain

        self._domains[domain] = state
        logger.info(
            "Ingestion complete for domain '%s': %d products, %d reviews",
            domain,
            len(normalized_products),
            len(reviews),
        )

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        """Query the recommendation system with natural language."""
        state = self._domains.get(domain)
        if state is None:
            logger.warning("Domain '%s' not ingested, returning empty results", domain)
            return []

        # Step 1: Parse query
        parsed = parse_query(query_text)

        # Step 2: Get all candidates (for small corpora, score everything)
        candidates = state.store.get_all_product_ids(domain)
        if not candidates:
            return []

        # Step 3: Build feature vectors
        features = state.assembler.build_features(parsed, candidates)

        # Step 4: Score with meta-ranker
        raw_scores = state.ranker.predict(features)

        # Step 5: Sigmoid normalization
        scores = normalize_scores(raw_scores)

        # Step 6: Get SHAP contributions for explanations
        contributions = state.ranker.predict_contributions(features)

        # Step 7: Rank and build results
        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in ranked_idx:
            pid = candidates[i]
            explanation, matched = build_explanation(contributions[i])
            product_name = state.store.get_product_name(pid)

            results.append(
                RecommendationResult(
                    product_id=pid,
                    product_name=product_name,
                    score=float(scores[i]),
                    explanation=explanation,
                    matched_attributes=matched,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_attributes(
        self,
        products: list[dict],
        reviews_by_product: dict[str, list[dict]],
        llm: LLMProvider,
    ) -> dict[str, list[dict]]:
        """Extract (aspect, sentiment, confidence) triples via LLM.

        Returns:
            {product_id: [{aspect, avg_sentiment, mention_count, confidence}, ...]}
        """
        extracted: dict[str, list[dict]] = {}

        for product in products:
            pid = product["product_id"]
            pname = product["product_name"]
            product_reviews = reviews_by_product.get(pid, [])

            if not product_reviews:
                extracted[pid] = []
                continue

            # Aggregate aspect mentions across all reviews for this product
            aspect_sentiments: dict[str, list[float]] = {}
            aspect_confidences: dict[str, list[float]] = {}

            for review in product_reviews:
                review_text = review.get("review_text", review.get("text", ""))
                if not review_text:
                    continue

                prompt = EXTRACTION_PROMPT.format(
                    product_name=pname,
                    review_text=review_text[:1500],
                )

                try:
                    response = llm.generate(prompt, json_mode=True)
                    parsed = json.loads(response)
                    attrs = parsed.get("attributes", [])

                    for attr in attrs:
                        aspect = attr.get("aspect", "").lower().strip()
                        if not aspect:
                            continue
                        sentiment = float(attr.get("sentiment", 0.0))
                        confidence = float(attr.get("confidence", 1.0))
                        aspect_sentiments.setdefault(aspect, []).append(sentiment)
                        aspect_confidences.setdefault(aspect, []).append(confidence)

                except (json.JSONDecodeError, KeyError, ValueError, RuntimeError) as e:
                    logger.warning(
                        "Failed to extract attributes from review for %s: %s",
                        pid,
                        e,
                    )
                    # Fallback: try to extract from product's static attributes
                    continue

            # Aggregate into per-product attribute scores
            aggregated = []
            for aspect in aspect_sentiments:
                sents = aspect_sentiments[aspect]
                confs = aspect_confidences.get(aspect, [1.0])
                aggregated.append({
                    "aspect": aspect,
                    "avg_sentiment": sum(sents) / len(sents),
                    "mention_count": len(sents),
                    "confidence": sum(confs) / len(confs),
                })

            # Also include static product attributes if available
            static_attrs = product.get("attributes", {})
            for attr_name, attr_val in static_attrs.items():
                if attr_name == "terrain":
                    continue
                if attr_name not in aspect_sentiments:
                    # Normalize 1-10 scale to -1 to 1
                    if isinstance(attr_val, (int, float)):
                        normalized = (attr_val - 5.0) / 5.0
                        aggregated.append({
                            "aspect": attr_name,
                            "avg_sentiment": normalized,
                            "mention_count": 1,
                            "confidence": 0.5,
                        })

            extracted[pid] = aggregated

        return extracted

    def _train_ranker(
        self,
        state: _DomainState,
        judgments: list[dict],
        products: list[dict],
        domain: str,
    ) -> None:
        """Train the XGBoost ranker from synthetic judgments."""
        feature_rows = []
        labels = []
        query_ids = []

        # Build a mapping for fast product lookup
        product_ids = state.store.get_all_product_ids(domain)
        product_id_set = set(product_ids)

        # Group judgments by query
        queries_seen: dict[str, int] = {}
        current_qid = 0

        for j in judgments:
            pid = j["product_id"]
            if pid not in product_id_set:
                continue

            query_text = j["query"]
            if query_text not in queries_seen:
                queries_seen[query_text] = current_qid
                current_qid += 1

            qid = queries_seen[query_text]
            parsed = parse_query(query_text)

            # Build feature vector for this (query, product) pair
            fv = state.assembler.build_features(parsed, [pid])
            if fv.shape[0] == 0:
                continue

            feature_rows.append(fv[0])
            labels.append(j["relevance"])
            query_ids.append(qid)

        if not feature_rows:
            logger.warning("No valid training samples, ranker will use fallback")
            return

        features = np.array(feature_rows, dtype=np.float32)
        label_arr = np.array(labels, dtype=np.float32)
        qid_arr = np.array(query_ids, dtype=np.int32)

        state.ranker.train(features, label_arr, qid_arr)


class _DomainState:
    """Per-domain state container."""

    def __init__(self) -> None:
        self.bm25_index: BM25Index | None = None
        self.faiss_index: FAISSIndex | None = None
        self.store: StructuredStore | None = None
        self.assembler: FeatureAssembler | None = None
        self.ranker: LTRRanker | None = None
        self.products: dict[str, dict] = {}
        self.domain: str = ""
