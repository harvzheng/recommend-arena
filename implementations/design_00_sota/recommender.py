"""Main Recommender for Design #0: SOTA Control (Retrieve-and-Rerank)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .index import ProductIndex
from .ingestion import build_product_record
from .query_parser import parse_query
from .scorer import score_and_rank
from .synonyms import expand_query, get_attribute_expansions
from .tfidf import TFIDFIndex

logger = logging.getLogger(__name__)


class SotaRecommender:
    """SOTA control: Retrieve-and-Rerank with ABSA extraction.

    Implements the shared Recommender protocol.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm or get_provider()
        self.index = ProductIndex()
        self.tfidf_index = TFIDFIndex()
        self._ingested_domains: set[str] = set()

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Ingest products and reviews for a domain.

        One LLM call per product (batched reviews) + one embedding per product.
        """
        if domain in self._ingested_domains:
            self.index.clear_domain(domain)
            self.tfidf_index.clear()
        self._ingested_domains.add(domain)

        # Group reviews by product
        reviews_by_product: dict[str, list[dict]] = {}
        for review in reviews:
            pid = review.get("product_id", "")
            if pid:
                reviews_by_product.setdefault(pid, []).append(review)

        # Use pre-defined attribute schema if available, fall back to inference
        from .index import DOMAIN_ATTRIBUTE_SCHEMAS
        if domain in DOMAIN_ATTRIBUTE_SCHEMAS:
            attribute_names = list(DOMAIN_ATTRIBUTE_SCHEMAS[domain])
        else:
            first_product = products[0] if products else {}
            attribute_names = [
                k for k, v in first_product.get("attributes", {}).items()
                if isinstance(v, (int, float))
            ]

        logger.info("Ingesting %d products for domain '%s' (attributes: %s)",
                     len(products), domain, attribute_names)

        for product in products:
            pid = product.get("product_id") or product.get("id", "")
            product_reviews = reviews_by_product.get(pid, [])

            record = build_product_record(
                product=product,
                reviews=product_reviews,
                domain=domain,
                llm=self.llm,
                attribute_names=attribute_names,
            )

            # Generate embedding
            try:
                record.embedding = self.llm.embed(record.review_text_combined)
            except Exception as e:
                logger.warning("Embedding failed for %s: %s", pid, e)

            self.index.add_product(record)

            # Build TF-IDF index from review text
            review_texts = [r.get("review_text") or r.get("text", "")
                           for r in product_reviews]
            combined_review_text = " ".join(review_texts)
            if combined_review_text.strip():
                self.tfidf_index.add_document(pid, combined_review_text)

        logger.info("Ingestion complete: %d products indexed for '%s'",
                     len(products), domain)

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        """Query with natural language, returns ranked recommendations."""
        logger.info("Query: '%s' (domain=%s, top_k=%d)", query_text, domain, top_k)

        # Step 0: Expand query with synonym dictionary BEFORE LLM parsing
        expanded_query = expand_query(query_text, domain)
        logger.info("Expanded query: '%s'", expanded_query)

        # Step 1: Parse query (using expanded text)
        parsed = parse_query(
            llm=self.llm,
            query_text=expanded_query,
            domain=domain,
            attribute_names=self.index.get_attribute_names(domain),
            spec_fields=self.index.get_spec_fields(domain),
            categories=self.index.get_categories(domain),
        )

        logger.info("Parsed: desired=%s, negative=%s, constraints=%s, categories=%s",
                     [a["name"] for a in parsed.desired_attributes],
                     [a["name"] for a in parsed.negative_attributes],
                     parsed.spec_constraints,
                     parsed.categories)

        # Step 2: Embed the query
        query_embedding = None
        try:
            query_embedding = self.llm.embed(parsed.query_embedding_text)
        except Exception as e:
            logger.warning("Query embedding failed: %s", e)

        # Step 3: Build TF-IDF query text (original + expanded attributes)
        attr_expansions = get_attribute_expansions(query_text, domain)
        tfidf_query_text = query_text + " " + " ".join(attr_expansions)

        # Step 4: Score and rank (with TF-IDF signal)
        scored = score_and_rank(
            index=self.index,
            parsed_query=parsed,
            query_embedding=query_embedding,
            domain=domain,
            top_k=top_k,
            tfidf_index=self.tfidf_index,
            tfidf_query_text=tfidf_query_text,
        )

        # Step 4: Normalize scores and build results
        if not scored:
            return []

        raw_scores = [sp.final_score for sp in scored]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        score_range = max_s - min_s

        results = []
        for sp in scored:
            if score_range > 0:
                norm_score = (sp.final_score - min_s) / score_range
            else:
                norm_score = 0.5

            norm_score = max(0.0, min(1.0, norm_score))

            explanation = _build_explanation(sp)

            results.append(RecommendationResult(
                product_id=sp.record.product_id,
                product_name=sp.record.product_name,
                score=round(norm_score, 4),
                explanation=explanation,
                matched_attributes=sp.matched_attributes,
            ))

        return results


def _build_explanation(sp) -> str:
    """Build explanation from matched attributes and review snippets."""
    parts = []
    name = sp.record.product_name

    if sp.matched_attributes:
        strong = [a for a, s in sp.matched_attributes.items() if s >= 0.7]
        moderate = [a for a, s in sp.matched_attributes.items() if 0.4 <= s < 0.7]
        weak = [a for a, s in sp.matched_attributes.items() if s < 0.4]

        if strong:
            parts.append(f"{name} excels in: {', '.join(strong)}.")
        if moderate:
            parts.append(f"Solid on: {', '.join(moderate)}.")
        if weak:
            parts.append(f"Weaker on: {', '.join(weak)}.")

    if sp.snippets:
        unique = list(dict.fromkeys(sp.snippets))[:3]
        for snippet in unique:
            parts.append(f'Review: "{snippet}"')

    if not parts:
        parts.append(f"{name} ({sp.record.category}).")

    return " ".join(parts)
