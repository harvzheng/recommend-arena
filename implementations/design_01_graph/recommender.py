"""Main Recommender implementation for Design #1: Graph-Based Knowledge Graph."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

# Ensure shared module is importable
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .config import get_domain_config
from .extraction import extract_aspects_from_review, extract_query_attributes
from .graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class GraphRecommender:
    """Graph-based recommendation system using NetworkX.

    Implements the shared Recommender protocol. Products, reviews, and
    attributes are modeled as nodes in a property graph. User queries are
    parsed into attribute targets and matched against the graph structure.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm or get_provider()
        self.kg = KnowledgeGraph()
        self._ingested_domains: set[str] = set()
        self._product_embeddings: dict[str, list[float]] = {}  # pid -> embedding vector

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Ingest product and review data for a given domain.

        Handles the benchmark data format where products have 'id'/'name'
        and reviews have 'product_id'/'text'.
        """
        logger.info("Ingesting %d products and %d reviews for domain '%s'",
                     len(products), len(reviews), domain)

        # Clear previous data for this domain if re-ingesting
        if domain in self._ingested_domains:
            self.kg.clear_domain(domain)
        self._ingested_domains.add(domain)

        domain_config = get_domain_config(domain)

        # --- 1. Add attribute nodes ---
        for attr_name in domain_config.get("attributes", {}):
            attr_id = f"attr-{domain}-{attr_name}"
            self.kg.add_attribute_node(attr_id, attr_name, domain)

        # --- 2. Add product nodes ---
        for product in products:
            pid = product.get("product_id") or product.get("id", "")
            pname = product.get("product_name") or product.get("name", "")
            category = product.get("category", "")
            specs = product.get("metadata") or product.get("specs", {})
            attributes = product.get("attributes", {})

            self.kg.add_product(
                product_id=pid,
                name=pname,
                domain=domain,
                category=category,
                specs=specs,
                attributes=attributes,
            )

            # Seed HAS_ATTRIBUTE edges from ground-truth attribute scores
            # This provides a baseline even without review extraction
            for attr_name, attr_value in attributes.items():
                if isinstance(attr_value, (int, float)):
                    attr_id = f"attr-{domain}-{attr_name}"
                    # Ensure the attribute node exists
                    self.kg.add_attribute_node(attr_id, attr_name, domain)
                    # Convert 1-10 scale to -1 to 1
                    normalized_score = (attr_value - 5.5) / 4.5
                    self.kg.set_has_attribute(
                        product_id=pid,
                        attr_id=attr_id,
                        score=normalized_score,
                        confidence=0.5,  # Moderate confidence for metadata
                        mention_count=0,
                    )

        # --- 3. Process reviews with LLM-based ABSA ---
        reviews_by_product: dict[str, list[dict]] = {}
        for i, review in enumerate(reviews):
            rid = review.get("review_id") or f"rev-{domain}-{i}"
            pid = review.get("product_id", "")
            text = review.get("review_text") or review.get("text", "")
            source = review.get("source", "")
            reviewer = review.get("reviewer", "")
            rating = review.get("rating")

            if not text or not pid:
                continue

            self.kg.add_review(
                review_id=rid,
                product_id=pid,
                text=text,
                domain=domain,
                source=source,
                reviewer=reviewer,
                rating=rating,
            )

            reviews_by_product.setdefault(pid, []).append({
                "review_id": rid,
                "text": text,
            })

        # Extract aspects from reviews
        for pid, rev_list in reviews_by_product.items():
            for rev in rev_list:
                try:
                    aspects = extract_aspects_from_review(
                        rev["text"], domain, self.llm
                    )
                except Exception as e:
                    logger.warning("Extraction failed for review %s: %s", rev["review_id"], e)
                    aspects = []

                for aspect in aspects:
                    attr_name = aspect["attribute"]
                    attr_id = f"attr-{domain}-{attr_name}"
                    # Ensure attribute node exists (may be newly discovered)
                    self.kg.add_attribute_node(attr_id, attr_name, domain)
                    self.kg.add_mention(
                        review_id=rev["review_id"],
                        attr_id=attr_id,
                        sentiment=aspect["sentiment"],
                        snippet=aspect.get("snippet", ""),
                    )

        # --- 4. Aggregate mentions into HAS_ATTRIBUTE edges ---
        for product in products:
            pid = product.get("product_id") or product.get("id", "")
            # Get all attribute nodes for this domain
            attr_nodes = [
                (n, d) for n, d in self.kg.graph.nodes(data=True)
                if d.get("type") == "attribute" and d.get("domain") == domain
            ]
            for attr_id, attr_data in attr_nodes:
                agg = self.kg.aggregate_has_attribute(pid, attr_id)
                if agg:
                    # Merge with existing ground-truth if present
                    existing = self.kg.get_has_attribute_edge(pid, attr_id)
                    if existing and existing.get("mention_count", 0) == 0:
                        # Ground truth only — blend with review data
                        gt_score = existing["score"]
                        review_score = agg["score"]
                        # Weight review data more heavily
                        blended = 0.3 * gt_score + 0.7 * review_score
                        confidence = agg["confidence"]
                    else:
                        blended = agg["score"]
                        confidence = agg["confidence"]

                    self.kg.set_has_attribute(
                        product_id=pid,
                        attr_id=attr_id,
                        score=blended,
                        confidence=confidence,
                        mention_count=agg["mention_count"],
                        snippets=agg.get("snippets", []),
                    )

        # --- 5. Generate product embeddings ---
        for product in products:
            pid = product.get("product_id") or product.get("id", "")
            pname = product.get("product_name") or product.get("name", "")
            category = product.get("category", "")

            # Build embedding text from product name + category + attributes + top snippets
            embed_parts = [pname, category]

            # Add attribute scores as descriptive text
            product_attrs = self.kg.get_product_attributes(pid)
            for attr_name, attr_data in product_attrs.items():
                score = attr_data.get("score", 0)
                if score > 0.3:
                    embed_parts.append(f"high {attr_name}")
                elif score < -0.3:
                    embed_parts.append(f"low {attr_name}")

                # Add top snippets
                for snippet in attr_data.get("snippets", [])[:2]:
                    embed_parts.append(snippet)

            embed_text = ". ".join(p for p in embed_parts if p)
            if embed_text:
                try:
                    self._product_embeddings[pid] = self.llm.embed(embed_text)
                except Exception as e:
                    logger.warning("Embedding failed for product %s: %s", pid, e)

        logger.info("Ingestion complete. Graph has %d nodes and %d edges, %d embeddings",
                     self.kg.graph.number_of_nodes(),
                     self.kg.graph.number_of_edges(),
                     len(self._product_embeddings))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        """Query the recommendation system with natural language."""
        logger.info("Query: '%s' (domain=%s, top_k=%d)", query_text, domain, top_k)

        # --- 1. Parse query ---
        parsed = extract_query_attributes(query_text, domain, self.llm)
        query_attrs = parsed.get("attributes", [])
        query_categories = parsed.get("categories", [])
        query_constraints = parsed.get("constraints", [])

        logger.info("Parsed query: attrs=%s, cats=%s, constraints=%s",
                     [a["attribute"] for a in query_attrs],
                     query_categories, query_constraints)

        # --- 2. Get candidates ---
        all_products = self.kg.get_products(domain)

        # Apply category filter
        if query_categories:
            filtered = []
            for pid in all_products:
                product_cat = self.kg.get_product_category(pid)
                if product_cat in query_categories:
                    filtered.append(pid)
                else:
                    # Also check terrain tags from ground truth
                    gt_attrs = self.kg.graph.nodes[pid].get("ground_truth_attributes", {})
                    terrain = gt_attrs.get("terrain", [])
                    # Check if any terrain maps to a query category
                    from .config import SKI_TERRAIN_MAP
                    for t in terrain:
                        mapped_cats = SKI_TERRAIN_MAP.get(t, [])
                        if any(qc in mapped_cats for qc in query_categories):
                            filtered.append(pid)
                            break
            # If filtering removes too many, keep all (soft filter)
            if len(filtered) >= 2:
                candidates = filtered
            else:
                candidates = all_products
        else:
            candidates = all_products

        # Apply spec constraints
        if query_constraints:
            constrained = []
            for pid in candidates:
                passes = True
                for c in query_constraints:
                    if not self.kg.check_constraint(pid, c["field"], c["op"], c["value"]):
                        passes = False
                        break
                if passes:
                    constrained.append(pid)
            # Soft filter — keep all if too few pass
            if len(constrained) >= 2:
                candidates = constrained

        # --- 3. Embed query for semantic scoring ---
        query_embedding: list[float] | None = None
        if self._product_embeddings:
            try:
                query_embedding = self.llm.embed(query_text)
            except Exception as e:
                logger.warning("Query embedding failed: %s", e)

        # --- 4. Score candidates ---
        scored: list[tuple[str, float, dict[str, float], list[str]]] = []
        for pid in candidates:
            if not query_attrs:
                # No attributes parsed — use embedding-only score if available
                if query_embedding and pid in self._product_embeddings:
                    embed_score = _cosine_similarity(query_embedding, self._product_embeddings[pid])
                    scored.append((pid, embed_score, {}, []))
                else:
                    scored.append((pid, 0.0, {}, []))
                continue

            raw_score, matched_attrs, snippets = self.kg.score_product(
                pid, query_attrs, domain
            )

            # Blend with embedding similarity
            if query_embedding and pid in self._product_embeddings:
                embed_sim = _cosine_similarity(query_embedding, self._product_embeddings[pid])
                # Blend: 0.7 graph + 0.3 embedding
                blended_score = 0.7 * raw_score + 0.3 * embed_sim
            else:
                blended_score = raw_score

            scored.append((pid, blended_score, matched_attrs, snippets))

        # Filter out products with no matches at all (unless too few remain)
        matched_products = [(pid, s, m, sn) for pid, s, m, sn in scored if m]
        if len(matched_products) >= min(top_k, 2):
            scored = matched_products

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # --- 5. Normalize scores and build results ---
        results: list[RecommendationResult] = []
        top_scored = scored[:top_k]

        if not top_scored:
            return []

        # Normalize from [-1, 1] to [0, 1]
        raw_scores = [s for _, s, _, _ in top_scored]
        min_score = min(raw_scores) if raw_scores else 0
        max_score = max(raw_scores) if raw_scores else 0
        score_range = max_score - min_score

        for pid, raw_score, matched_attrs, snippets in top_scored:
            product_data = self.kg.get_product_data(pid)

            # Normalize score to 0-1
            if score_range > 0:
                normalized_score = (raw_score - min_score) / score_range
            else:
                normalized_score = 0.5 if raw_score >= 0 else 0.0

            # Clamp to [0, 1]
            normalized_score = max(0.0, min(1.0, normalized_score))

            # Build explanation
            explanation = self._build_explanation(
                product_data, matched_attrs, snippets, query_attrs
            )

            results.append(RecommendationResult(
                product_id=pid,
                product_name=product_data.get("name", pid),
                score=round(normalized_score, 4),
                explanation=explanation,
                matched_attributes=matched_attrs,
            ))

        return results

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    def _build_explanation(
        self,
        product_data: dict,
        matched_attrs: dict[str, float],
        snippets: list[str],
        query_attrs: list[dict],
    ) -> str:
        """Build a human-readable explanation for a recommendation."""
        parts = []
        product_name = product_data.get("name", "This product")

        if matched_attrs:
            # Describe attribute matches
            strong_matches = [
                attr for attr, strength in matched_attrs.items() if strength >= 0.7
            ]
            weak_matches = [
                attr for attr, strength in matched_attrs.items() if 0.3 <= strength < 0.7
            ]
            poor_matches = [
                attr for attr, strength in matched_attrs.items() if strength < 0.3
            ]

            if strong_matches:
                parts.append(
                    f"{product_name} strongly matches on: {', '.join(strong_matches)}."
                )
            if weak_matches:
                parts.append(
                    f"Moderate match on: {', '.join(weak_matches)}."
                )
            if poor_matches:
                parts.append(
                    f"Weak match on: {', '.join(poor_matches)}."
                )

        # Add review snippets as evidence
        if snippets:
            unique_snippets = list(dict.fromkeys(snippets))[:3]  # Deduplicate
            for snippet in unique_snippets:
                parts.append(f'Review: "{snippet}"')

        if not parts:
            category = product_data.get("category", "")
            parts.append(f"{product_name} ({category}).")

        return " ".join(parts)
