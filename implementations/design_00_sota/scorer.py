"""Hybrid scoring: semantic recall + structured attributes + TF-IDF re-ranking."""

from __future__ import annotations

from dataclasses import dataclass, field

from .index import ProductIndex, ProductRecord
from .query_parser import ParsedQuery
from .tfidf import TFIDFIndex


@dataclass
class ScoredProduct:
    """A product with its computed scores and explanation data."""
    record: ProductRecord
    semantic_score: float = 0.0
    attribute_score: float = 0.0
    tfidf_score: float = 0.0
    negation_penalty: float = 0.0
    final_score: float = 0.0
    matched_attributes: dict[str, float] = field(default_factory=dict)
    snippets: list[str] = field(default_factory=list)


# Weights for combining signals
SEMANTIC_WEIGHT = 0.25
ATTRIBUTE_WEIGHT = 0.55
TFIDF_WEIGHT = 0.20


def score_and_rank(
    index: ProductIndex,
    parsed_query: ParsedQuery,
    query_embedding: list[float] | None,
    domain: str,
    top_k: int = 10,
    tfidf_index: TFIDFIndex | None = None,
    tfidf_query_text: str = "",
) -> list[ScoredProduct]:
    """Score all candidates and return top_k ranked results."""
    candidates = index.get_domain_products(domain)

    # Stage 1: Category filter (soft — keep all if too few match)
    if parsed_query.categories:
        filtered = [p for p in candidates if p.category in parsed_query.categories]
        if len(filtered) >= 2:
            candidates = filtered

    # Stage 2: Spec constraint filter (soft)
    if parsed_query.spec_constraints:
        constrained = [p for p in candidates if _passes_constraints(p, parsed_query.spec_constraints)]
        if len(constrained) >= 2:
            candidates = constrained

    # Pre-compute TF-IDF similarities for all candidates
    tfidf_scores: dict[str, float] = {}
    if tfidf_index and tfidf_query_text:
        tfidf_scores = tfidf_index.query_all_similarities(tfidf_query_text)

    # Stage 3: Score each candidate
    scored = []
    for product in candidates:
        sp = ScoredProduct(record=product)

        # Semantic score from embedding similarity
        if query_embedding and product.embedding:
            sp.semantic_score = index.cosine_similarity(query_embedding, product.embedding)

        # Attribute score from structured data (with near-miss decay)
        sp.attribute_score, sp.matched_attributes, sp.snippets = _compute_attribute_score(
            product, parsed_query
        )

        # TF-IDF keyword score
        sp.tfidf_score = tfidf_scores.get(product.product_id, 0.0)

        # Negation penalty
        sp.negation_penalty = _compute_negation_penalty(product, parsed_query)

        # Combine all three signals
        sp.final_score = (
            SEMANTIC_WEIGHT * sp.semantic_score
            + ATTRIBUTE_WEIGHT * sp.attribute_score
            + TFIDF_WEIGHT * sp.tfidf_score
            - sp.negation_penalty
        )

        scored.append(sp)

    scored.sort(key=lambda x: x.final_score, reverse=True)
    return scored[:top_k]


def _get_attribute_value(product: ProductRecord, attr_name: str) -> float | None:
    """Get attribute value, preferring ABSA-extracted over ground truth.

    Blends review-extracted scores with ground truth when both exist.
    Returns value on 0-10 scale, or None if not found.
    """
    review_data = product.review_attributes.get(attr_name)
    gt_value = product.ground_truth_attributes.get(attr_name)

    if isinstance(gt_value, list):
        gt_value = None  # terrain lists etc. aren't numeric

    if review_data and gt_value is not None:
        review_score = review_data["score"]
        confidence = review_data.get("confidence", 0.5)
        # Blend: higher confidence in reviews → weight reviews more
        return confidence * review_score + (1 - confidence) * float(gt_value)
    elif review_data:
        return review_data["score"]
    elif gt_value is not None:
        return float(gt_value)
    return None


def _compute_attribute_score(
    product: ProductRecord,
    parsed_query: ParsedQuery,
) -> tuple[float, dict[str, float], list[str]]:
    """Compute weighted attribute match score.

    Returns (score, matched_attributes, snippets).
    """
    if not parsed_query.desired_attributes:
        return 0.0, {}, []

    total_weight = 0.0
    weighted_score = 0.0
    matched = {}
    snippets = []

    for attr in parsed_query.desired_attributes:
        name = attr["name"]
        weight = attr["weight"]
        direction = attr.get("direction", "high")
        total_weight += weight

        value = _get_attribute_value(product, name)
        if value is None:
            continue

        # Near-miss decay scoring (exponential reward near target)
        norm = value / 10.0
        norm = max(0.0, min(1.0, norm))
        if direction == "high":
            # Exponential: score = norm^2 — sharply rewards high values
            match_strength = norm ** 2
        else:
            # Low direction: score = (1 - norm)^2
            match_strength = (1.0 - norm) ** 2

        weighted_score += weight * match_strength
        matched[name] = round(match_strength, 3)

        # Collect snippets from ABSA data
        review_data = product.review_attributes.get(name)
        if review_data and review_data.get("snippets"):
            snippets.extend(review_data["snippets"][:2])

    if total_weight == 0:
        return 0.0, matched, snippets

    return weighted_score / total_weight, matched, snippets


def _compute_negation_penalty(
    product: ProductRecord,
    parsed_query: ParsedQuery,
) -> float:
    """Compute penalty for attributes the user explicitly doesn't want."""
    if not parsed_query.negative_attributes:
        return 0.0

    total_penalty = 0.0
    total_neg_weight = 0.0
    for attr in parsed_query.negative_attributes:
        name = attr["name"]
        weight = attr["weight"]
        total_neg_weight += weight

        value = _get_attribute_value(product, name)
        if value is None:
            continue

        # High value on a negated attribute = big penalty
        norm = value / 10.0
        total_penalty += weight * norm * 0.5  # Scale penalty to not overwhelm

    if total_neg_weight == 0:
        return 0.0
    return total_penalty / total_neg_weight


def _passes_constraints(product: ProductRecord, constraints: list[dict]) -> bool:
    """Check if a product passes all spec constraints."""
    for c in constraints:
        field_name = c["field"]
        op = c["op"]
        target = c["value"]

        if target is None:
            continue

        spec_value = product.specs.get(field_name)
        if spec_value is None:
            continue  # No data — don't penalize

        # Handle list specs (e.g., lengths_cm)
        if isinstance(spec_value, list):
            values = [v for v in spec_value if v is not None]
            if not values:
                continue
            if op == ">=":
                if not any(v >= target for v in values):
                    return False
            elif op == "<=":
                if not any(v <= target for v in values):
                    return False
            elif op == "==":
                if target not in values:
                    return False
        else:
            if op == ">=" and spec_value < target:
                return False
            elif op == "<=" and spec_value > target:
                return False
            elif op == "==" and spec_value != target:
                return False

    return True
