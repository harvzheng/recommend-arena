"""Tests for vectors.py — chunking, vector construction, centroids."""

import numpy as np

from implementations.design_11_finetuned_embed.vectors import (
    AttributeCentroid,
    ProductVector,
    build_attribute_centroids,
    chunk_review,
)


def test_chunk_review_basic():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_review(text, max_sentences=3)
    assert len(chunks) >= 1
    assert "First sentence." in chunks[0]


def test_chunk_review_skips_short():
    text = "Hi. Ok. Sure."
    chunks = chunk_review(text, max_sentences=3)
    for chunk in chunks:
        assert len(chunk.split()) >= 5 or len(chunks) == 0


def test_chunk_review_empty():
    assert chunk_review("") == []
    assert chunk_review("   ") == []


def test_chunk_review_single_long_sentence():
    text = "This is a single long sentence that should appear as one passage."
    chunks = chunk_review(text, max_sentences=3)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_product_vector_shape():
    vec = np.random.randn(384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    pv = ProductVector(
        product_id="SKI-001",
        product_name="Test Ski",
        vector=vec,
        passage_vectors=np.random.randn(3, 384).astype(np.float32),
        passage_texts=["a", "b", "c"],
        attribute_scores={"stiffness": 9},
    )
    assert pv.vector.shape == (384,)
    assert pv.passage_vectors.shape == (3, 384)


def test_build_attribute_centroids():
    rng = np.random.RandomState(42)
    pvecs = []
    for i in range(4):
        v = rng.randn(384).astype(np.float32)
        v = v / np.linalg.norm(v)
        stiffness_score = 9 if i < 3 else 2
        pvecs.append(ProductVector(
            product_id=f"SKI-{i:03d}",
            product_name=f"Ski {i}",
            vector=v,
            passage_vectors=rng.randn(2, 384).astype(np.float32),
            passage_texts=["passage a", "passage b"],
            attribute_scores={"stiffness": stiffness_score, "playfulness": 10 - stiffness_score},
        ))

    centroids = build_attribute_centroids(pvecs, ["stiffness", "playfulness"], threshold=7.0)

    attr_names = {c.attribute for c in centroids}
    assert "stiffness" in attr_names, "Should build stiffness centroid (3 high products)"

    for c in centroids:
        assert c.centroid.shape == (384,)
        assert abs(np.linalg.norm(c.centroid) - 1.0) < 1e-5


def test_build_attribute_centroids_skips_insufficient():
    rng = np.random.RandomState(42)
    v = rng.randn(384).astype(np.float32)
    v = v / np.linalg.norm(v)
    pvecs = [ProductVector(
        product_id="SKI-001",
        product_name="Ski 1",
        vector=v,
        passage_vectors=rng.randn(1, 384).astype(np.float32),
        passage_texts=["passage"],
        attribute_scores={"stiffness": 9},
    )]

    centroids = build_attribute_centroids(pvecs, ["stiffness"], threshold=7.0)
    assert len(centroids) == 0, "Should skip when < 2 products above threshold"
