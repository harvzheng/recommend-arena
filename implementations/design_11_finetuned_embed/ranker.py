"""Cosine ranking and explanation generation for Design #11."""

from __future__ import annotations

import numpy as np

from .vectors import AttributeCentroid, ProductVector


def rank_products(
    query_text: str,
    product_vectors: list[ProductVector],
    model,
    top_k: int = 10,
) -> list[tuple[ProductVector, float]]:
    """Rank products by cosine similarity to query in fine-tuned space."""
    query_vec = model.encode(query_text, normalize_embeddings=True)

    product_matrix = np.stack([pv.vector for pv in product_vectors])

    scores = product_matrix @ query_vec

    min_s, max_s = float(scores.min()), float(scores.max())
    score_range = max_s - min_s if max_s > min_s else 1.0
    normalized = (scores - min_s) / score_range

    ranked_indices = np.argsort(-normalized)[:top_k]
    return [
        (product_vectors[i], float(normalized[i]))
        for i in ranked_indices
    ]


def explain_match(
    query_vec: np.ndarray,
    product_vec: np.ndarray,
    centroids: list[AttributeCentroid],
) -> tuple[str, dict[str, float]]:
    """Decompose query-product similarity into per-attribute contributions."""
    contributions: dict[str, float] = {}

    for centroid in centroids:
        query_alignment = float(query_vec @ centroid.centroid)
        product_alignment = float(product_vec @ centroid.centroid)
        contribution = max(0.0, query_alignment * product_alignment)
        if contribution > 0.01:
            contributions[centroid.attribute] = round(contribution, 3)

    total = sum(contributions.values()) or 1.0
    matched_attributes = {
        attr: round(score / total, 3)
        for attr, score in sorted(contributions.items(), key=lambda x: -x[1])
    }

    if matched_attributes:
        top_attrs = list(matched_attributes.items())[:3]
        attr_parts = [f"{attr} ({score:.0%})" for attr, score in top_attrs]
        explanation = f"Strong match on {', '.join(attr_parts)}."
        if len(matched_attributes) > 3:
            explanation += f" Also relevant: {', '.join(a for a, _ in list(matched_attributes.items())[3:])}."
    else:
        explanation = "General semantic match (no specific attribute alignment detected)."

    return explanation, matched_attributes
