"""IDF-style attribute weighting.

Attributes that appear in every product (e.g. every ski has a stiffness
rating) get lower IDF weight; rare attributes (e.g. "touring binding
compatible") get higher IDF.  This naturally emphasises discriminative
attributes.
"""

from __future__ import annotations

import math


def compute_idf_weights(
    product_vectors: list[dict[str, float]],
    threshold: float = 0.1,
) -> dict[str, float]:
    """Compute IDF weights across all product feature vectors.

    Args:
        product_vectors: List of sparse feature dicts ``{key: value}``.
        threshold: Minimum absolute value to consider an attribute
            "present" in a product.

    Returns:
        Dict mapping feature keys to IDF weight (>= 0).
    """
    n_products = len(product_vectors)
    if n_products == 0:
        return {}

    # Collect all keys
    all_keys: set[str] = set()
    for pv in product_vectors:
        all_keys.update(pv.keys())

    idf: dict[str, float] = {}
    for key in all_keys:
        doc_freq = sum(
            1 for pv in product_vectors if abs(pv.get(key, 0.0)) > threshold
        )
        if doc_freq == 0:
            idf[key] = 0.0
        else:
            # Standard IDF with +1 smoothing (avoids zero for universal attrs)
            idf[key] = math.log(n_products / doc_freq) + 1.0

    return idf
