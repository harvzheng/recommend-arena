# Design #11: Fine-Tuned Embedding Ranker

## 1. Architecture Overview

This design makes the embedding space itself the recommendation engine. Instead of relying on a general-purpose encoder and hoping it captures domain-specific attribute gradients -- "stiff" vs "soft," "damp" vs "chattery" -- we contrastive-train a small transformer so that the geometry of the vector space directly reflects structured attribute magnitudes. A query for "stiff carving ski" lands closer to products with `stiffness: 9` than `stiffness: 4` not because of lexical overlap or downstream reranking, but because the model was trained to place them there.

The core bet: a 22M-parameter model fine-tuned on a few thousand attribute-driven contrastive pairs will outperform a 335M-parameter off-the-shelf encoder (Design #2's BGE-large) for domain-specific retrieval, at 47x fewer parameters and no query-time LLM calls.

### Core Flow

```
                    TRAINING (offline, once per domain)
                    =================================

Structured Product Scores          Review Passages
(stiffness: 8, edge_grip: 9)      ("locked-in on hardpack...")
         |                                  |
         v                                  v
  ┌──────────────────────────────────────────────┐
  │  Contrastive Pair Generator                  │
  │  anchor: "ski with high stiffness"           │
  │  positive: passage from product scoring 9    │
  │  negative: passage from product scoring 3    │
  └──────────────────┬───────────────────────────┘
                     |
                     v
  ┌──────────────────────────────────────────────┐
  │  Fine-Tune all-MiniLM-L6-v2                  │
  │  MultipleNegativesRankingLoss                │
  │  5-10 epochs, batch 16, lr 2e-5              │
  └──────────────────┬───────────────────────────┘
                     |
                     v
              Trained Encoder (90MB)


                    INFERENCE (per query)
                    ====================

  User Query: "stiff damp carving ski"
         |
         v
  ┌──────────────┐
  │ Trained       │──> query vector (384-dim)
  │ Encoder       │        |
  └──────────────┘         v
                    ┌──────────────────┐
                    │ Cosine Sim vs     │
                    │ Product Vectors   │──> ranked results
                    │ (numpy, in-mem)   │
                    └──────────────────┘
                           |
                           v
                    ┌──────────────────┐
                    │ Centroid-Based    │
                    │ Explanation       │──> per-attribute contributions
                    └──────────────────┘
```

At query time there is one encoder forward pass (~5ms on CPU), a matrix multiply against 25-50 product vectors, and a centroid decomposition for explanations. No LLM calls. No reranker. No vector database. The model IS the system.

### Why This Might Work

Off-the-shelf encoders treat "very stiff" and "somewhat stiff" as nearly synonymous because they share tokens and context. After contrastive training with products that have `stiffness: 9` as positives and `stiffness: 3` as negatives, the model learns that "very stiff" and "somewhat stiff" should be geometrically distant -- they belong to different neighborhoods. This is a fundamentally different inductive bias than pre-trained encoders provide.

### Why It Might Not

The training signal comes from structured product scores cross-referenced with review passages. If the structured scores are noisy (a product rated `stiffness: 7` when reviewers consistently describe it as "medium flex"), the model learns the wrong geometry. The system is only as good as the attribute labels it trains on.

Additionally, with only 25 products and ~135 reviews in the benchmark dataset, there is a real risk of overfitting -- the model memorizes specific passages rather than learning generalizable attribute gradients. The 20% validation holdout is essential, and we may need early stopping.

---

## 2. Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Base encoder** | `all-MiniLM-L6-v2` (22M params, 384-dim) | Small enough to fine-tune on CPU in minutes. Proven architecture for sentence similarity. 384-dim is plenty for 25-50 products. |
| **Fine-tuning** | `sentence-transformers` built-in trainers | First-class support for contrastive learning. `MultipleNegativesRankingLoss` handles in-batch negatives automatically. No custom training loop needed. |
| **Loss function** | `MultipleNegativesRankingLoss` | Standard for contrastive pair training. In-batch negatives give O(batch^2) training signal from O(batch) examples. |
| **Vector index** | `numpy` in-memory cosine similarity | 25-50 products, 384-dim. A matrix multiply takes <1ms. ANN indexing is overkill. |
| **Storage** | JSON (product data) + pickle (model, vectors) | No database. Serialize the fine-tuned model and pre-computed product vectors. Total on-disk: ~95MB. |
| **Pair generation** | Template-based anchors + passage selection | Deterministic, reproducible. No LLM calls during pair generation. |
| **Training data augmentation** | None for training; synthetic available for eval | Real data only during training. The benchmark harness can generate synthetic queries for evaluation. |
| **Explanations** | Per-attribute centroid decomposition | Compute attribute centroids from high-scoring products, then project query-product similarity onto centroid directions. |

### Why MiniLM Over BGE

Design #2 uses BGE-large (335M params, 1024-dim) because it bets on pre-trained quality. This design bets on domain-adapted quality from a much smaller model. MiniLM-L6-v2 is the right choice because:

1. **Fine-tuning speed.** 22M params trains in 2-5 minutes on CPU. BGE-large would take 30-60 minutes and realistically needs a GPU.
2. **Inference speed.** 384-dim forward pass is ~5ms on CPU vs ~25ms for BGE-large. At query time, this is the only compute we do.
3. **Overfitting risk.** With a small training set (~2-5K pairs from 25 products), a smaller model is less likely to memorize.
4. **Sufficient capacity.** We are learning a mapping from ~8 attribute dimensions to a 384-dim space. The model has more than enough capacity for this.

---

## 3. Data Model

### 3.1 Training Data Structures

```python
from dataclasses import dataclass, field


@dataclass
class AttributeScore:
    """A single attribute score for a product."""
    attribute: str      # e.g. "stiffness", "edge_grip", "dampness"
    score: float        # 1-10 scale from structured product data


@dataclass
class ProductProfile:
    """Product with structured attribute scores and associated review passages."""
    product_id: str
    product_name: str
    domain: str
    attribute_scores: dict[str, float]       # attribute -> 1-10 score
    review_passages: list[str]               # chunked review text, 1-3 sentences each
    metadata: dict = field(default_factory=dict)


@dataclass
class ContrastivePair:
    """A single training example: anchor text paired with a review passage.

    The training framework handles in-batch negatives, so we only need
    (anchor, positive) pairs. Other positives in the same batch serve as
    negatives.
    """
    anchor: str         # template-generated: "ski with high stiffness"
    positive: str       # review passage from a product scoring high on that attribute


@dataclass
class TrainingConfig:
    """Hyperparameters for fine-tuning."""
    base_model: str = "all-MiniLM-L6-v2"
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_fraction: float = 0.10
    validation_split: float = 0.20
    output_dir: str = "models/design-11"
    seed: int = 42
```

### 3.2 Runtime Data Structures

```python
import numpy as np


@dataclass
class ProductVector:
    """A product's representation in the fine-tuned embedding space."""
    product_id: str
    product_name: str
    vector: np.ndarray          # 384-dim, L2-normalized, mean of passage embeddings
    passage_vectors: np.ndarray  # (n_passages, 384), for explanation generation
    passage_texts: list[str]
    attribute_scores: dict[str, float]  # original structured scores, for evaluation


@dataclass
class AttributeCentroid:
    """Centroid vector for products scoring high on a specific attribute.

    Used during explanation generation to decompose query-product similarity
    into per-attribute contributions.
    """
    attribute: str
    centroid: np.ndarray    # 384-dim, L2-normalized
    high_scoring_ids: list[str]  # product_ids with score >= 7 on this attribute
```

---

## 4. Training Pipeline

Training is the core differentiator. This section describes how structured product scores become a contrastive training signal that reshapes the embedding space.

### 4.1 Pair Construction

The pair generator creates (anchor, positive) examples by crossing template-based anchor texts with review passages from products that score high on the relevant attribute. `MultipleNegativesRankingLoss` treats other in-batch positives as negatives, so we do not explicitly construct negative pairs.

```python
import random
from itertools import product as iterproduct


# Anchor templates per attribute. Multiple phrasings prevent the model
# from memorizing a single template rather than learning the concept.
ANCHOR_TEMPLATES: dict[str, list[str]] = {
    "stiffness": [
        "ski with high stiffness",
        "very stiff ski",
        "rigid demanding ski",
        "ski that is extremely stiff and firm",
    ],
    "edge_grip": [
        "ski with excellent edge grip",
        "ski that holds on ice",
        "strong edge hold on hardpack",
        "ski with incredible bite on firm snow",
    ],
    "dampness": [
        "ski with great vibration dampening",
        "very damp and composed ski",
        "smooth and damp ski at speed",
        "ski that absorbs chatter well",
    ],
    "stability": [
        "very stable ski at high speed",
        "ski that feels planted and locked in",
        "rock solid stability",
        "ski with confidence-inspiring stability",
    ],
    "playfulness": [
        "playful and fun ski",
        "lively and energetic ski",
        "ski that is easy to pivot and smear",
        "poppy and responsive ski",
    ],
    "weight": [
        "lightweight ski",
        "very light ski for touring",
        "featherweight ski",
        "ski that is impressively light",
    ],
    "turn_initiation": [
        "ski with easy turn initiation",
        "ski that turns effortlessly",
        "quick and responsive turn entry",
        "ski that snaps into turns",
    ],
    "versatility": [
        "versatile all-mountain ski",
        "do-it-all ski for any condition",
        "ski that handles everything well",
        "quiver-of-one ski",
    ],
}

# Thresholds for selecting positive passages
HIGH_SCORE_THRESHOLD = 7   # products scoring >= 7 provide positive passages
LOW_SCORE_THRESHOLD = 4    # products scoring <= 4 would be negatives (handled by in-batch)


def generate_contrastive_pairs(
    products: list[ProductProfile],
    min_pairs_per_attribute: int = 50,
    seed: int = 42,
) -> list[ContrastivePair]:
    """Generate (anchor, positive) pairs from structured scores and review passages.

    For each attribute:
      1. Find products scoring >= HIGH_SCORE_THRESHOLD.
      2. For each such product, pair its review passages with anchor templates.
      3. Shuffle and balance so each attribute contributes roughly equally.

    Returns ~500-3000 pairs depending on attribute count and review density.
    """
    rng = random.Random(seed)
    all_pairs: list[ContrastivePair] = []

    for attribute, templates in ANCHOR_TEMPLATES.items():
        attr_pairs: list[ContrastivePair] = []

        # Find high-scoring products for this attribute
        high_products = [
            p for p in products
            if p.attribute_scores.get(attribute, 0) >= HIGH_SCORE_THRESHOLD
        ]

        if not high_products:
            continue

        for prod in high_products:
            for passage in prod.review_passages:
                # Skip very short passages (likely not opinion-bearing)
                if len(passage.split()) < 8:
                    continue
                # Pair with a randomly chosen anchor template
                anchor = rng.choice(templates)
                attr_pairs.append(ContrastivePair(anchor=anchor, positive=passage))

        # Ensure minimum representation per attribute
        if len(attr_pairs) < min_pairs_per_attribute and attr_pairs:
            # Oversample with different anchor templates
            while len(attr_pairs) < min_pairs_per_attribute:
                base = rng.choice(attr_pairs)
                new_anchor = rng.choice(templates)
                attr_pairs.append(ContrastivePair(anchor=new_anchor, positive=base.positive))

        rng.shuffle(attr_pairs)
        all_pairs.extend(attr_pairs)

    rng.shuffle(all_pairs)
    return all_pairs
```

**Why template-based anchors instead of LLM-generated?** Templates are deterministic, fast, and reproducible. The model does not need creative anchor phrasing -- it needs consistent signal about which direction in embedding space corresponds to which attribute. Four templates per attribute (with minor lexical variation) are sufficient to prevent template memorization.

**Why not review-product pairs?** We considered using query-like review sentences as anchors (e.g., "this ski is incredibly stiff" paired with the product vector). The problem: review sentences describe what a product IS, not what a user WANTS. The embedding space should be organized by user intent, not product description. Template anchors simulate user queries.

### 4.2 Fine-Tuning

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader


def fine_tune(
    pairs: list[ContrastivePair],
    config: TrainingConfig,
) -> SentenceTransformer:
    """Fine-tune MiniLM on contrastive pairs.

    Uses MultipleNegativesRankingLoss which treats other in-batch positives
    as negatives. With batch_size=16, each example gets 15 implicit negatives
    per step -- enough contrastive signal for our small dataset.
    """
    model = SentenceTransformer(config.base_model)

    # Split into train/val
    split_idx = int(len(pairs) * (1 - config.validation_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Convert to InputExample format
    train_examples = [
        InputExample(texts=[p.anchor, p.positive]) for p in train_pairs
    ]

    train_loader = DataLoader(
        train_examples,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Loss: in-batch negatives contrastive loss
    loss = losses.MultipleNegativesRankingLoss(model)

    # Validation evaluator (optional but useful for monitoring)
    val_sentences_1 = [p.anchor for p in val_pairs]
    val_sentences_2 = [p.positive for p in val_pairs]
    val_scores = [1.0] * len(val_pairs)  # all val pairs are positives

    evaluator = EmbeddingSimilarityEvaluator(
        val_sentences_1, val_sentences_2, val_scores,
        name="attribute-val",
    )

    # Train
    warmup_steps = int(len(train_loader) * config.epochs * config.warmup_fraction)

    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=config.epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=len(train_loader),  # evaluate once per epoch
        output_path=config.output_dir,
        optimizer_params={"lr": config.learning_rate},
        show_progress_bar=True,
    )

    return model
```

### 4.3 Training Budget

With 8 attributes, 25 products, and ~5.4 reviews per product (135 total), the pair counts work out to:

| Step | Count |
|------|-------|
| Products scoring >= 7 per attribute (avg) | ~8 |
| Review passages per product (avg, after chunking) | ~12 |
| Raw pairs per attribute | ~96 |
| After oversampling to min 50 | ~96 (no oversampling needed) |
| Total pairs (8 attributes) | ~768 |
| After 20% val holdout | ~614 train, ~154 val |
| Training steps per epoch (batch 16) | ~38 |
| Total training steps (8 epochs) | ~304 |
| Estimated training time (CPU) | 2-5 minutes |
| Estimated training time (GPU) | 15-30 seconds |

This is a very small training run. The risk is underfitting (not enough signal) rather than computational cost. If validation loss plateaus early, we can increase epochs to 15-20 without meaningful time cost.

### 4.4 Model Export

```python
import pickle
from pathlib import Path


def export_model(model: SentenceTransformer, config: TrainingConfig) -> Path:
    """Save the fine-tuned model for inference.

    The sentence-transformers library saves the full model directory.
    We also pickle the config for reproducibility.
    """
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # sentence-transformers saves during .fit(), but we can also save explicitly
    model.save(str(output / "model"))

    # Save config for reproducibility
    with open(output / "training_config.json", "w") as f:
        import json
        json.dump(vars(config), f, indent=2)

    return output
```

---

## 5. Ingestion Pipeline

Ingestion builds product vector representations using the fine-tuned encoder. Unlike Design #2 (which stores per-passage vectors in ChromaDB), this design compresses each product into a single 384-dim vector -- the mean of its review passage embeddings.

```
Raw Products + Reviews
        |
        v
[1] Chunk reviews into passages (1-3 sentences)
        |
        v
[2] Encode all passages with fine-tuned model
        |
        v
[3] Mean-pool passage vectors per product, L2-normalize
        |
        v
[4] Compute attribute centroids (for explanations)
        |
        v
[5] Serialize to disk (JSON + pickle)
```

### 5.1 Passage Chunking

Same sentence-boundary chunking as Design #2, but simpler: no opinion filtering needed. The fine-tuned model has learned to attend to attribute-relevant content and downweight logistics ("shipped fast"). Filtering is baked into the geometry.

```python
import re


def chunk_review(review_text: str, max_sentences: int = 3) -> list[str]:
    """Split review into overlapping passages of 1-3 sentences."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', review_text) if s.strip()]
    if not sentences:
        return []

    passages = []
    for i in range(0, len(sentences), max_sentences - 1):  # 1-sentence overlap
        chunk = " ".join(sentences[i:i + max_sentences])
        if len(chunk.split()) >= 5:  # skip very short fragments
            passages.append(chunk)
    return passages
```

### 5.2 Product Vector Construction

```python
def build_product_vectors(
    products: list[dict],
    reviews: list[dict],
    model: SentenceTransformer,
) -> list[ProductVector]:
    """Build a single vector per product from review passage embeddings.

    Each product vector is the L2-normalized mean of its passage embeddings.
    This works because the fine-tuned space is organized by attribute gradients --
    averaging passages preserves the product's position along each attribute axis.
    """
    # Group reviews by product
    reviews_by_product: dict[str, list[str]] = {}
    for review in reviews:
        pid = review["product_id"]
        if pid not in reviews_by_product:
            reviews_by_product[pid] = []
        reviews_by_product[pid].append(review["review_text"])

    product_vectors = []
    for product in products:
        pid = product["product_id"]
        review_texts = reviews_by_product.get(pid, [])

        # Chunk all reviews into passages
        all_passages = []
        for text in review_texts:
            all_passages.extend(chunk_review(text))

        if not all_passages:
            # No reviews: use product name + metadata as fallback
            fallback = f"{product['product_name']} {product.get('domain', '')}"
            all_passages = [fallback]

        # Encode all passages
        passage_vecs = model.encode(
            all_passages, normalize_embeddings=True, show_progress_bar=False,
        )

        # Mean pool and re-normalize
        mean_vec = passage_vecs.mean(axis=0)
        mean_vec = mean_vec / np.linalg.norm(mean_vec)

        product_vectors.append(ProductVector(
            product_id=pid,
            product_name=product["product_name"],
            vector=mean_vec,
            passage_vectors=passage_vecs,
            passage_texts=all_passages,
            attribute_scores=product.get("attributes", {}),
        ))

    return product_vectors
```

### 5.3 Attribute Centroid Construction

Centroids are pre-computed during ingestion so explanation generation at query time is free (no additional encoding needed).

```python
def build_attribute_centroids(
    product_vectors: list[ProductVector],
    attributes: list[str],
    threshold: float = 7.0,
) -> list[AttributeCentroid]:
    """Build per-attribute centroids from high-scoring products.

    For each attribute, the centroid is the mean (L2-normalized) of product
    vectors where that attribute scores >= threshold. These centroids define
    "directions" in embedding space that correspond to attribute concepts.
    """
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
            # Not enough products to form a meaningful centroid. Skip.
            continue

        centroid = np.mean(high_vecs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        centroids.append(AttributeCentroid(
            attribute=attr,
            centroid=centroid,
            high_scoring_ids=high_ids,
        ))

    return centroids
```

### 5.4 Serialization

```python
import json
import pickle
from pathlib import Path


def save_index(
    product_vectors: list[ProductVector],
    centroids: list[AttributeCentroid],
    output_dir: str,
) -> None:
    """Persist the vector index and centroids to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Product vectors: pickle for numpy arrays, JSON for metadata
    with open(out / "product_vectors.pkl", "wb") as f:
        pickle.dump(product_vectors, f)

    # Centroids
    with open(out / "attribute_centroids.pkl", "wb") as f:
        pickle.dump(centroids, f)

    # Human-readable product index (for debugging)
    index = [
        {
            "product_id": pv.product_id,
            "product_name": pv.product_name,
            "n_passages": len(pv.passage_texts),
            "attribute_scores": pv.attribute_scores,
        }
        for pv in product_vectors
    ]
    with open(out / "product_index.json", "w") as f:
        json.dump(index, f, indent=2)
```

---

## 6. Query / Ranking Pipeline

Query-time is deliberately minimal. One forward pass through the encoder, one matrix multiply for cosine similarity, done. The fine-tuned geometry does all the heavy lifting.

```
User Query: "stiff damp carving ski for hardpack"
        |
        v
[1] Encode query with fine-tuned model --> 384-dim vector (~5ms CPU)
        |
        v
[2] Cosine similarity vs all product vectors --> scores (<1ms)
        |
        v
[3] Min-max normalize to [0, 1]
        |
        v
[4] Generate explanations via centroid decomposition
        |
        v
[5] Return top-K RecommendationResult
```

### 6.1 Ranking Implementation

```python
def rank_products(
    query_text: str,
    product_vectors: list[ProductVector],
    model: SentenceTransformer,
    top_k: int = 10,
) -> list[tuple[ProductVector, float]]:
    """Rank products by cosine similarity to the query in fine-tuned space.

    This is the entire ranking algorithm. No reranker, no LLM, no post-processing.
    The embedding space was trained to make this work.
    """
    # Encode query
    query_vec = model.encode(query_text, normalize_embeddings=True)

    # Build product matrix (N x 384)
    product_matrix = np.stack([pv.vector for pv in product_vectors])

    # Cosine similarity (vectors are already L2-normalized)
    scores = product_matrix @ query_vec  # (N,)

    # Min-max normalize to [0, 1]
    min_s, max_s = scores.min(), scores.max()
    score_range = max_s - min_s if max_s > min_s else 1.0
    normalized = (scores - min_s) / score_range

    # Sort and return top-K
    ranked_indices = np.argsort(-normalized)[:top_k]
    return [
        (product_vectors[i], float(normalized[i]))
        for i in ranked_indices
    ]
```

### 6.2 Why No Reranker?

Design #2 uses a cross-encoder reranker on top of BGE retrieval. We skip it because:

1. **The retrieval IS the ranking.** In Design #2, BGE retrieval is a recall-oriented first stage that needs precision refinement. Here, the fine-tuned model is trained to be the scoring function, not just a recall filter.
2. **Cross-encoders add latency and complexity.** A MiniLM cross-encoder on top-30 candidates adds ~150ms. Our total query time is ~6ms.
3. **Cross-encoders are also pre-trained.** A cross-encoder's domain knowledge is no better than the off-the-shelf bi-encoder it reranks. Fine-tuning the bi-encoder directly is more principled than stacking another pre-trained model on top.

If the fine-tuned model proves insufficient (e.g., it ranks "medium stiff" products above "very stiff" products for a "stiff ski" query), the correct fix is better training data, not a reranker band-aid.

### 6.3 Handling Queries That Fall Outside Training Distribution

The fine-tuned model retains MiniLM's pre-trained knowledge for concepts not covered by fine-tuning. A query about "price" or "brand" will fall back to general semantic similarity, which is reasonable but not domain-optimized. This is acceptable: the system is designed for attribute-based queries, not metadata lookups.

```python
# Queries that work well (trained attributes):
# "stiff carving ski"              -> stiffness gradient
# "damp and stable at speed"       -> dampness + stability gradients
# "lightweight touring ski"        -> weight gradient

# Queries that work okay (general semantics):
# "best ski for intermediate"      -> picks up skill-level language in reviews
# "good value for money"           -> pre-trained understanding of "value"

# Queries that don't work (metadata, not attributes):
# "cheapest ski under $500"        -> needs structured metadata filtering
# "Nordica brand skis"             -> needs metadata, not embedding similarity
```

---

## 7. Explanation Generation

Explanations decompose the query-product similarity into per-attribute contributions using the pre-computed attribute centroids. The intuition: if a product scores high because it is close to the query along the "stiffness direction" in embedding space, the stiffness centroid will have high dot product with both the query and the product.

### 7.1 Centroid Decomposition

```python
def explain_match(
    query_vec: np.ndarray,
    product_vec: np.ndarray,
    centroids: list[AttributeCentroid],
) -> tuple[str, dict[str, float]]:
    """Decompose query-product similarity into per-attribute contributions.

    For each attribute centroid C_a:
      contribution_a = (query . C_a) * (product . C_a)

    This measures how much the query and product agree along the attribute
    direction. High values mean both the query asks for this attribute and
    the product delivers it.

    Returns:
        explanation: Human-readable string.
        matched_attributes: Dict of attribute -> contribution score (0-1 normalized).
    """
    contributions: dict[str, float] = {}

    for centroid in centroids:
        query_alignment = float(query_vec @ centroid.centroid)
        product_alignment = float(product_vec @ centroid.centroid)
        # Both alignments are cosine similarities (normalized vectors), range [-1, 1].
        # Product of two cosine sims: range [-1, 1]. We clamp to [0, 1].
        contribution = max(0.0, query_alignment * product_alignment)
        if contribution > 0.01:  # skip negligible contributions
            contributions[centroid.attribute] = round(contribution, 3)

    # Normalize contributions to sum to 1 (relative importance)
    total = sum(contributions.values()) or 1.0
    matched_attributes = {
        attr: round(score / total, 3)
        for attr, score in sorted(contributions.items(), key=lambda x: -x[1])
    }

    # Build explanation string
    if matched_attributes:
        top_attrs = list(matched_attributes.items())[:3]
        attr_parts = [f"{attr} ({score:.0%})" for attr, score in top_attrs]
        explanation = f"Strong match on {', '.join(attr_parts)}."
        if len(matched_attributes) > 3:
            explanation += f" Also relevant: {', '.join(a for a, _ in list(matched_attributes.items())[3:])}."
    else:
        explanation = "General semantic match (no specific attribute alignment detected)."

    return explanation, matched_attributes
```

### 7.2 Explanation Quality

The centroid decomposition is approximate. It assumes attribute directions in embedding space are roughly orthogonal, which they may not be after fine-tuning (stiffness and stability might be correlated). For a precise decomposition, we would need to orthogonalize the centroid basis, but for human-readable explanations, the approximate version is more intuitive: users understand "strong match on stiffness (45%), stability (30%)" even if those contributions are not perfectly additive.

The key property we preserve: if a product ranks high because of stiffness, the stiffness contribution will be visibly high. The explanations are directionally correct even if not mathematically exact.

---

## 8. Benchmark Integration

### 8.1 Recommender Protocol Implementation

```python
from sentence_transformers import SentenceTransformer

from shared.interface import Recommender, RecommendationResult


class FineTunedEmbeddingRecommender:
    """Design #11: Fine-tuned embedding ranker.

    The embedding space IS the recommendation engine. Query-time cost is
    one encoder forward pass + cosine similarity. No LLM calls.
    """

    def __init__(self, model_dir: str | None = None):
        self.model: SentenceTransformer | None = None
        self.model_dir = model_dir
        self.product_vectors: dict[str, list[ProductVector]] = {}   # domain -> vectors
        self.centroids: dict[str, list[AttributeCentroid]] = {}     # domain -> centroids
        self._trained = False

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Ingest products and reviews: train (if needed) and build index.

        If a pre-trained model exists at self.model_dir, load it.
        Otherwise, train from scratch using structured product scores.
        """
        # Step 1: Build product profiles (structured scores + review passages)
        profiles = self._build_profiles(products, reviews, domain)

        # Step 2: Train or load the fine-tuned model
        if self.model is None:
            if self.model_dir and Path(self.model_dir).exists():
                self.model = SentenceTransformer(str(Path(self.model_dir) / "model"))
                self._trained = True
            else:
                self.model = self._train(profiles)
                self._trained = True

        # Step 3: Build product vectors
        pvecs = build_product_vectors(products, reviews, self.model)
        self.product_vectors[domain] = pvecs

        # Step 4: Build attribute centroids
        attributes = list(ANCHOR_TEMPLATES.keys())
        self.centroids[domain] = build_attribute_centroids(pvecs, attributes)

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        """Query the fine-tuned embedding space."""
        if domain not in self.product_vectors:
            return []

        pvecs = self.product_vectors[domain]
        centroids = self.centroids.get(domain, [])

        # Encode query
        query_vec = self.model.encode(query_text, normalize_embeddings=True)

        # Rank by cosine similarity
        ranked = rank_products(query_text, pvecs, self.model, top_k=top_k)

        # Build results with explanations
        results = []
        for pv, score in ranked:
            explanation, matched_attrs = explain_match(
                query_vec, pv.vector, centroids
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
        """Convert raw dicts to ProductProfile objects for training."""
        reviews_by_product: dict[str, list[str]] = {}
        for r in reviews:
            pid = r["product_id"]
            if pid not in reviews_by_product:
                reviews_by_product[pid] = []
            reviews_by_product[pid].append(r["review_text"])

        profiles = []
        for p in products:
            pid = p["product_id"]
            passages = []
            for text in reviews_by_product.get(pid, []):
                passages.extend(chunk_review(text))

            profiles.append(ProductProfile(
                product_id=pid,
                product_name=p["product_name"],
                domain=domain,
                attribute_scores=p.get("metadata", {}),
                review_passages=passages,
                metadata=p.get("metadata", {}),
            ))
        return profiles

    def _train(self, profiles: list[ProductProfile]) -> SentenceTransformer:
        """Train the fine-tuned model from product profiles."""
        config = TrainingConfig()
        pairs = generate_contrastive_pairs(profiles)
        model = fine_tune(pairs, config)
        export_model(model, config)
        return model
```

### 8.2 Benchmark Compatibility

The `FineTunedEmbeddingRecommender` satisfies the `Recommender` protocol:

```python
from shared.interface import Recommender

rec = FineTunedEmbeddingRecommender()

# Verify protocol compliance
assert isinstance(rec, Recommender)

# Standard benchmark flow
rec.ingest(products=catalog, reviews=reviews, domain="ski")
results = rec.query("stiff damp carving ski for hardpack", domain="ski", top_k=10)

for r in results:
    assert 0.0 <= r.score <= 1.0
    assert isinstance(r.matched_attributes, dict)
    assert isinstance(r.explanation, str)
    print(f"{r.product_name}: {r.score:.3f}")
    print(f"  {r.explanation}")
    print(f"  Attributes: {r.matched_attributes}")
```

### 8.3 First-Run Behavior

On first `ingest()`, the system trains the model (2-5 minutes on CPU). Subsequent runs in the same process reuse the trained model. To persist across processes, pass `model_dir` pointing to a saved checkpoint:

```python
# First run: trains and saves
rec = FineTunedEmbeddingRecommender(model_dir="models/design-11")
rec.ingest(products, reviews, domain="ski")  # trains, saves to models/design-11/

# Second run: loads saved model
rec2 = FineTunedEmbeddingRecommender(model_dir="models/design-11")
rec2.ingest(products, reviews, domain="ski")  # loads, skips training
```

### 8.4 LLM Usage

This design uses the shared `LLMProvider` from `shared/llm_provider.py` only during training pair construction if synthetic anchor augmentation is enabled (not in the default configuration). At query time, zero LLM calls are made. This is a deliberate design constraint: the model should be self-contained after training.

| Phase | LLM Calls | Embedding Model Calls |
|-------|-----------|----------------------|
| Pair generation | 0 (template-based) | 0 |
| Fine-tuning | 0 | ~600 batches (internal to training loop) |
| Ingestion | 0 | 1 batch per product (~25 calls) |
| Query | 0 | 1 call (encode query) |

---

## 9. Trade-offs & Limitations

### Strengths

- **Query-time simplicity.** One forward pass + one matrix multiply. No LLM calls, no reranker, no multi-stage pipeline. Latency is ~6ms total on CPU.
- **Small model, big signal.** 22M parameters vs 335M (Design #2) or multiple models (Design #10). Ships as a single 90MB file.
- **Ordinal gradients are learned, not approximated.** Unlike Design #2's contrastive scoring (which uses pos/neg query pairs as a workaround for embedding magnitude blindness), this design trains the magnitude sensitivity directly into the space.
- **No infrastructure dependencies.** No vector database, no external APIs at query time, no GPU required. Pure Python + numpy.
- **Interpretable via centroid decomposition.** Explanations trace back to specific attribute directions in the trained space, not black-box similarity scores.
- **Fast iteration.** Training takes 2-5 minutes. Changing the attribute schema or adding new training signal is cheap.

### Weaknesses

- **Training data dependency.** The structured product scores (stiffness: 1-10, etc.) must exist and be accurate. If the scores are noisy or missing for some attributes, the trained geometry is wrong in those directions. Design #2 needs no structured data at all.
- **Overfitting risk with small datasets.** 25 products, ~768 training pairs, 22M parameters. The model may memorize specific review phrasings rather than learning generalizable attribute gradients. The 20% validation holdout mitigates but does not eliminate this risk.
- **Domain-specific training required.** Adding a new domain (running shoes) means defining new anchor templates and re-training. Design #2 handles new domains with zero additional work. This is the price of domain adaptation.
- **No cross-domain transfer.** A model trained on ski attributes will not understand running shoe attributes. Each domain needs its own fine-tuned model.
- **Metadata queries are unsupported.** "Cheapest ski" or "Nordica brand" queries cannot be answered by embedding similarity. These require structured metadata filtering (not implemented in this design).
- **Cold start is worse than Design #2.** Design #2 can ingest a product with zero reviews and still retrieve it by name/description. This design's product vector from a single fallback passage will be poor quality.
- **Centroid explanations are approximate.** Attribute directions in the trained space are not orthogonal. Contributions may overcount when attributes are correlated (stiffness and stability often co-occur in carving skis).

### Honest Assessment: Where This Design Wins and Loses

**Wins against Design #2 (Pure Embedding):** On queries targeting specific attributes with ordinal preferences ("stiffest carving ski," "lightest touring ski"), the fine-tuned model should reliably rank products by attribute magnitude. Design #2's contrastive scoring is a clever hack but cannot match learned gradients.

**Loses against Design #2 (Pure Embedding):** On novel or vague queries ("a ski that inspires confidence," "something fun for spring slush"), the pre-trained BGE model has broader semantic coverage. Fine-tuning narrows the model's focus, potentially degrading performance on queries outside the trained attribute space.

**Wins against Design #8 (TF-IDF):** No LLM calls at query time. No attribute catalog to maintain. No synonym tables. The model learns its own attribute vocabulary from the training data.

**Loses against Design #8 (TF-IDF):** Design #8's structured attribute matching is perfectly interpretable and debuggable. This design's centroid explanations are approximate. Design #8 also handles negation ("no rocker") naturally through negative feature weights; negation in embedding space is poorly defined.

**Wins against Design #10 (Ensemble):** Simpler architecture (one model vs five + XGBoost). Faster query time (6ms vs ~500ms). Fewer moving parts to break.

**Loses against Design #10 (Ensemble):** Design #10 can combine structured and unstructured signals. If the fine-tuned embedding misses something, there is no second-chance retrieval. Design #10's XGBoost ranker can learn complex non-linear interactions between features that a linear embedding space cannot represent.

---

## 10. Future Directions

### 10.1 Hard Negative Mining

The current training uses only in-batch negatives (random). Hard negative mining would select negatives that are close in embedding space but wrong on the target attribute -- e.g., a product with `stiffness: 7` as a negative when the anchor asks for `stiffness: 9`. This sharpens the gradient boundary between adjacent ordinal values.

```python
def mine_hard_negatives(
    model: SentenceTransformer,
    products: list[ProductProfile],
    attribute: str,
    anchor: str,
    k_hard: int = 3,
) -> list[str]:
    """Find passages that are close to the anchor but from low-scoring products."""
    anchor_vec = model.encode(anchor, normalize_embeddings=True)
    low_products = [p for p in products if p.attribute_scores.get(attribute, 5) <= LOW_SCORE_THRESHOLD]

    candidates = []
    for p in low_products:
        for passage in p.review_passages:
            pvec = model.encode(passage, normalize_embeddings=True)
            sim = float(anchor_vec @ pvec)
            candidates.append((passage, sim))

    # Return the closest (hardest) negatives
    candidates.sort(key=lambda x: -x[1])
    return [c[0] for c in candidates[:k_hard]]
```

### 10.2 Multi-Domain Models

Instead of training separate models per domain, train a single model with domain-prefixed anchors: "ski with high stiffness," "running shoe with high cushioning." If attribute vocabularies overlap (both domains have "weight," "responsiveness"), the model could learn transferable gradients. This is speculative -- it may also cause interference between domains.

### 10.3 Online Adaptation

After deployment, user click-through data provides a new training signal. Products that users select after querying are implicit positives; products shown but not selected are implicit negatives. Periodic re-training (or LoRA-style adapter updates) can continuously improve the model.

### 10.4 Matryoshka Representations

MiniLM supports Matryoshka representation learning, where the first N dimensions of the embedding are a valid lower-dimensional representation. Training with Matryoshka loss would let us use 128-dim or even 64-dim vectors for ultra-fast retrieval with minimal quality loss. At 25 products this is unnecessary, but at 10K+ products the storage and compute savings matter.

### 10.5 Attribute-Specific Subspaces

Instead of a single 384-dim space encoding all attributes, partition the dimensions: dims 0-47 for stiffness, 48-95 for edge_grip, etc. This guarantees orthogonal attribute directions and makes centroid decomposition exact. The cost: each attribute gets only ~48 dimensions of capacity, which may be insufficient for nuanced concepts.

### 10.6 Synthetic Training Data via LLM

Use the shared `LLMProvider` to generate synthetic review passages for underrepresented attribute combinations. For example, if no product scores `stiffness: 2` AND `edge_grip: 9`, generate plausible review text for that combination. This fills gaps in the training distribution without requiring more real products. The risk is that synthetic text has different distributional properties than real reviews, introducing a domain shift in the training data.
