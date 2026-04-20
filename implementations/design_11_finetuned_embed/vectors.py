"""Product vector construction and centroid computation for Design #11."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ProductVector:
    """A product's representation in the fine-tuned embedding space."""
    product_id: str
    product_name: str
    vector: np.ndarray
    passage_vectors: np.ndarray
    passage_texts: list[str]
    attribute_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class AttributeCentroid:
    """Centroid vector for products scoring high on a specific attribute."""
    attribute: str
    centroid: np.ndarray
    high_scoring_ids: list[str]


def chunk_review(review_text: str, max_sentences: int = 3) -> list[str]:
    """Split review into overlapping passages of 1-3 sentences."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', review_text) if s.strip()]
    if not sentences:
        return []

    passages = []
    for i in range(0, len(sentences), max_sentences - 1):
        chunk = " ".join(sentences[i:i + max_sentences])
        if len(chunk.split()) >= 5:
            passages.append(chunk)
    return passages


def build_product_vectors(
    products: list[dict],
    reviews: list[dict],
    model,
) -> list[ProductVector]:
    """Build a single vector per product from review passage embeddings."""
    reviews_by_product: dict[str, list[str]] = {}
    for review in reviews:
        pid = review.get("product_id", review.get("id", ""))
        text = review.get("review_text", review.get("text", ""))
        if pid and text:
            if pid not in reviews_by_product:
                reviews_by_product[pid] = []
            reviews_by_product[pid].append(text)

    product_vectors = []
    for product in products:
        pid = product.get("product_id", product.get("id", ""))
        pname = product.get("product_name", product.get("name", ""))

        review_texts = reviews_by_product.get(pid, [])

        all_passages = []
        for text in review_texts:
            all_passages.extend(chunk_review(text))

        if not all_passages:
            fallback = f"{pname} {product.get('domain', '')}"
            all_passages = [fallback]

        passage_vecs = model.encode(
            all_passages, normalize_embeddings=True, show_progress_bar=False,
        )

        mean_vec = passage_vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        product_vectors.append(ProductVector(
            product_id=pid,
            product_name=pname,
            vector=mean_vec,
            passage_vectors=passage_vecs,
            passage_texts=all_passages,
            attribute_scores=product.get("attributes", {}),
        ))

    return product_vectors


def build_attribute_centroids(
    product_vectors: list[ProductVector],
    attributes: list[str],
    threshold: float = 7.0,
) -> list[AttributeCentroid]:
    """Build per-attribute centroids from high-scoring products."""
    centroids = []
    for attr in attributes:
        high_vecs = []
        high_ids = []
        for pv in product_vectors:
            score = pv.attribute_scores.get(attr, 0)
            if isinstance(score, (int, float)) and score >= threshold:
                high_vecs.append(pv.vector)
                high_ids.append(pv.product_id)

        if len(high_vecs) < 2:
            continue

        centroid = np.mean(high_vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        centroids.append(AttributeCentroid(
            attribute=attr,
            centroid=centroid,
            high_scoring_ids=high_ids,
        ))

    return centroids


def save_index(
    product_vectors: list[ProductVector],
    centroids: list[AttributeCentroid],
    output_dir: str,
) -> None:
    """Persist the vector index and centroids to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "product_vectors.pkl", "wb") as f:
        pickle.dump(product_vectors, f)

    with open(out / "attribute_centroids.pkl", "wb") as f:
        pickle.dump(centroids, f)

    index = [
        {
            "product_id": pv.product_id,
            "product_name": pv.product_name,
            "n_passages": len(pv.passage_texts),
            "attribute_scores": {
                k: v for k, v in pv.attribute_scores.items()
                if isinstance(v, (int, float))
            },
        }
        for pv in product_vectors
    ]
    with open(out / "product_index.json", "w") as f:
        json.dump(index, f, indent=2)


def load_index(
    input_dir: str,
) -> tuple[list[ProductVector], list[AttributeCentroid]]:
    """Load a previously saved vector index and centroids."""
    inp = Path(input_dir)

    with open(inp / "product_vectors.pkl", "rb") as f:
        product_vectors = pickle.load(f)

    with open(inp / "attribute_centroids.pkl", "rb") as f:
        centroids = pickle.load(f)

    return product_vectors, centroids
