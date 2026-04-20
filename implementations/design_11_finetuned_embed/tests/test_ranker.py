"""Tests for ranker.py — ranking order and explanation generation."""

import numpy as np

from implementations.design_11_finetuned_embed.ranker import explain_match
from implementations.design_11_finetuned_embed.vectors import (
    AttributeCentroid,
    ProductVector,
)


def _normed(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_explain_match_returns_top_attributes():
    dim = 384
    rng = np.random.RandomState(42)

    query_vec = _normed(rng.randn(dim).astype(np.float32))
    product_vec = _normed(query_vec + 0.1 * rng.randn(dim).astype(np.float32))

    stiffness_centroid = _normed(query_vec + 0.05 * rng.randn(dim).astype(np.float32))
    orth = rng.randn(dim).astype(np.float32)
    orth = orth - (orth @ query_vec) * query_vec
    playfulness_centroid = _normed(orth)

    centroids = [
        AttributeCentroid(attribute="stiffness", centroid=stiffness_centroid, high_scoring_ids=["A"]),
        AttributeCentroid(attribute="playfulness", centroid=playfulness_centroid, high_scoring_ids=["B"]),
    ]

    explanation, matched = explain_match(query_vec, product_vec, centroids)

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert isinstance(matched, dict)
    if "stiffness" in matched and "playfulness" in matched:
        assert matched["stiffness"] >= matched["playfulness"]


def test_explain_match_empty_centroids():
    dim = 384
    rng = np.random.RandomState(42)
    query_vec = _normed(rng.randn(dim).astype(np.float32))
    product_vec = _normed(rng.randn(dim).astype(np.float32))

    explanation, matched = explain_match(query_vec, product_vec, [])

    assert "General semantic match" in explanation
    assert matched == {}


def test_explain_match_contains_attribute_names():
    dim = 384
    rng = np.random.RandomState(42)

    base = _normed(rng.randn(dim).astype(np.float32))
    query_vec = _normed(base + 0.01 * rng.randn(dim).astype(np.float32))
    product_vec = _normed(base + 0.01 * rng.randn(dim).astype(np.float32))
    centroid_vec = _normed(base + 0.01 * rng.randn(dim).astype(np.float32))

    centroids = [
        AttributeCentroid(attribute="edge_grip", centroid=centroid_vec, high_scoring_ids=["A", "B"]),
    ]

    explanation, matched = explain_match(query_vec, product_vec, centroids)

    assert "edge_grip" in matched or "edge_grip" in explanation


def test_ranking_order_with_mock_model():
    dim = 384
    rng = np.random.RandomState(42)

    query_direction = _normed(rng.randn(dim).astype(np.float32))

    vec_a = _normed(query_direction + 0.05 * rng.randn(dim).astype(np.float32))
    vec_b = _normed(rng.randn(dim).astype(np.float32))

    pv_a = ProductVector(
        product_id="A", product_name="Close Ski", vector=vec_a,
        passage_vectors=rng.randn(1, dim).astype(np.float32),
        passage_texts=["close"], attribute_scores={"stiffness": 9},
    )
    pv_b = ProductVector(
        product_id="B", product_name="Far Ski", vector=vec_b,
        passage_vectors=rng.randn(1, dim).astype(np.float32),
        passage_texts=["far"], attribute_scores={"stiffness": 3},
    )

    class MockModel:
        def encode(self, text, normalize_embeddings=True, show_progress_bar=False):
            return query_direction

    from implementations.design_11_finetuned_embed.ranker import rank_products

    ranked = rank_products("stiff ski", [pv_a, pv_b], MockModel(), top_k=2)
    assert ranked[0][0].product_id == "A", "Product closer to query should rank first"
    assert ranked[0][1] >= ranked[1][1], "Scores should be descending"
