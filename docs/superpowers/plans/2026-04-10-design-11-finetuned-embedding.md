# Design #11: Fine-Tuned Embedding Ranker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a contrastive fine-tuned embedding ranker that trains all-MiniLM-L6-v2 on attribute-driven pairs and ranks products by cosine similarity in the learned space.

**Architecture:** Fine-tune a small sentence transformer on contrastive pairs derived from structured product attribute scores. At query time, encode the query, cosine-rank against pre-computed product vectors, and explain matches via attribute centroid decomposition. Zero LLM calls at query time.

**Tech Stack:** sentence-transformers, numpy, torch, pytest

---

## File Structure

```
implementations/design_11_finetuned_embed/
├── __init__.py           # factory: create_recommender()
├── requirements.txt      # sentence-transformers, numpy, torch
├── pairs.py              # ContrastivePair generation from attribute scores
├── trainer.py            # Fine-tuning wrapper around sentence-transformers
├── vectors.py            # Product vector construction + centroid computation
├── ranker.py             # Cosine ranking + explanation generation
├── recommender.py        # Main class implementing Recommender protocol
└── tests/
    ├── __init__.py
    ├── test_pairs.py     # Test pair generation
    ├── test_vectors.py   # Test vector construction
    ├── test_ranker.py    # Test ranking + explanations
    └── test_recommender.py  # Integration test
```

Each file has one job:
- `pairs.py` — dataclasses (AttributeScore, ProductProfile, ContrastivePair, TrainingConfig) and contrastive pair generation
- `trainer.py` — fine-tuning and model export using sentence-transformers
- `vectors.py` — review chunking, product vector construction, centroid computation, serialization
- `ranker.py` — cosine ranking and centroid-based explanation generation
- `recommender.py` — wires pairs → trainer → vectors → ranker behind Recommender protocol

---

### Task 1: Data Models and Pair Generation

**Files:**
- Create: `implementations/design_11_finetuned_embed/pairs.py`
- Create: `implementations/design_11_finetuned_embed/tests/__init__.py`
- Create: `implementations/design_11_finetuned_embed/tests/test_pairs.py`

- [ ] **Step 1.1: Create `pairs.py` with dataclasses**

```python
"""Data models and contrastive pair generation for Design #11."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class AttributeScore:
    """A single attribute score for a product."""
    attribute: str
    score: float


@dataclass
class ProductProfile:
    """Product with structured attribute scores and associated review passages."""
    product_id: str
    product_name: str
    domain: str
    attribute_scores: dict[str, float]
    review_passages: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class ContrastivePair:
    """A single training example: anchor text paired with a review passage.

    The training framework handles in-batch negatives, so we only need
    (anchor, positive) pairs.
    """
    anchor: str
    positive: str


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


HIGH_SCORE_THRESHOLD = 7
LOW_SCORE_THRESHOLD = 4

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
    "damp": [
        "ski with great vibration dampening",
        "very damp and composed ski",
        "smooth and damp ski at speed",
        "ski that absorbs chatter well",
    ],
    "stability_at_speed": [
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
    "powder_float": [
        "ski with great powder float",
        "ski that floats in deep snow",
        "powder ski with effortless float",
        "ski that surfs through powder",
    ],
    "forgiveness": [
        "forgiving and easy ski",
        "ski that is very forgiving of mistakes",
        "easy and approachable ski",
        "ski that doesn't punish errors",
    ],
    "versatility": [
        "versatile all-mountain ski",
        "do-it-all ski for any condition",
        "ski that handles everything well",
        "quiver-of-one ski",
    ],
}
```

- [ ] **Step 1.2: Add `generate_contrastive_pairs()` to `pairs.py`**

Append this function to the end of `pairs.py`:

```python
def generate_contrastive_pairs(
    products: list[ProductProfile],
    min_pairs_per_attribute: int = 50,
    seed: int = 42,
) -> list[ContrastivePair]:
    """Generate (anchor, positive) pairs from structured scores and review passages.

    For each attribute, find products scoring >= HIGH_SCORE_THRESHOLD, then
    pair their review passages with anchor templates. Oversample if needed
    to reach min_pairs_per_attribute.
    """
    rng = random.Random(seed)
    all_pairs: list[ContrastivePair] = []

    for attribute, templates in ANCHOR_TEMPLATES.items():
        attr_pairs: list[ContrastivePair] = []

        high_products = [
            p for p in products
            if p.attribute_scores.get(attribute, 0) >= HIGH_SCORE_THRESHOLD
        ]

        if not high_products:
            continue

        for prod in high_products:
            for passage in prod.review_passages:
                if len(passage.split()) < 8:
                    continue
                anchor = rng.choice(templates)
                attr_pairs.append(ContrastivePair(anchor=anchor, positive=passage))

        if len(attr_pairs) < min_pairs_per_attribute and attr_pairs:
            while len(attr_pairs) < min_pairs_per_attribute:
                base = rng.choice(attr_pairs)
                new_anchor = rng.choice(templates)
                attr_pairs.append(ContrastivePair(anchor=new_anchor, positive=base.positive))

        rng.shuffle(attr_pairs)
        all_pairs.extend(attr_pairs)

    rng.shuffle(all_pairs)
    return all_pairs
```

- [ ] **Step 1.3: Create `tests/__init__.py`**

```python
```

(Empty file.)

- [ ] **Step 1.4: Create `tests/test_pairs.py` and run tests**

```python
"""Tests for pairs.py — dataclasses and contrastive pair generation."""

from implementations.design_11_finetuned_embed.pairs import (
    AttributeScore,
    ContrastivePair,
    ProductProfile,
    TrainingConfig,
    ANCHOR_TEMPLATES,
    HIGH_SCORE_THRESHOLD,
    generate_contrastive_pairs,
)


def test_attribute_score_construction():
    s = AttributeScore(attribute="stiffness", score=9.0)
    assert s.attribute == "stiffness"
    assert s.score == 9.0


def test_product_profile_construction():
    p = ProductProfile(
        product_id="SKI-001",
        product_name="Test Ski",
        domain="ski",
        attribute_scores={"stiffness": 9},
        review_passages=["This ski is incredibly stiff and demanding."],
    )
    assert p.product_id == "SKI-001"
    assert p.review_passages[0].startswith("This ski")


def test_contrastive_pair_construction():
    pair = ContrastivePair(anchor="very stiff ski", positive="Incredibly rigid.")
    assert pair.anchor == "very stiff ski"


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.base_model == "all-MiniLM-L6-v2"
    assert cfg.epochs == 8
    assert cfg.batch_size == 16
    assert cfg.learning_rate == 2e-5


def _make_products() -> list[ProductProfile]:
    """Two products: one high stiffness, one low."""
    return [
        ProductProfile(
            product_id="SKI-001",
            product_name="Stiff Ski",
            domain="ski",
            attribute_scores={"stiffness": 9, "playfulness": 2},
            review_passages=[
                "This ski is incredibly stiff and demanding on hardpack.",
                "Rigid and powerful, locks into carving turns with precision.",
                "Very firm underfoot, no flex at all even at high speed.",
            ],
        ),
        ProductProfile(
            product_id="SKI-002",
            product_name="Soft Ski",
            domain="ski",
            attribute_scores={"stiffness": 3, "playfulness": 8},
            review_passages=[
                "Super playful and fun ski that is easy to throw around.",
                "Lively and energetic, great for park laps and butters.",
                "Very forgiving soft flex, perfect for learning new tricks.",
            ],
        ),
    ]


def test_generate_pairs_returns_pairs():
    products = _make_products()
    pairs = generate_contrastive_pairs(products, min_pairs_per_attribute=5, seed=42)
    assert len(pairs) > 0
    assert all(isinstance(p, ContrastivePair) for p in pairs)


def test_generate_pairs_covers_attributes():
    products = _make_products()
    pairs = generate_contrastive_pairs(products, min_pairs_per_attribute=5, seed=42)
    anchors = {p.anchor for p in pairs}
    # Should have stiffness anchors (product 1 scores 9)
    stiffness_templates = set(ANCHOR_TEMPLATES["stiffness"])
    assert anchors & stiffness_templates, "Expected stiffness anchors in output"
    # Should have playfulness anchors (product 2 scores 8)
    play_templates = set(ANCHOR_TEMPLATES["playfulness"])
    assert anchors & play_templates, "Expected playfulness anchors in output"


def test_generate_pairs_positive_from_high_scoring():
    products = _make_products()
    pairs = generate_contrastive_pairs(products, min_pairs_per_attribute=5, seed=42)
    stiffness_templates = set(ANCHOR_TEMPLATES["stiffness"])
    stiff_pairs = [p for p in pairs if p.anchor in stiffness_templates]
    # Positives for stiffness should come from SKI-001 (score 9), not SKI-002 (score 3)
    stiff_passages = {p.positive for p in stiff_pairs}
    ski1_passages = set(products[0].review_passages)
    assert stiff_passages <= ski1_passages, "Stiffness positives should come from high-scoring product"


def test_generate_pairs_skips_short_passages():
    products = [
        ProductProfile(
            product_id="SKI-X",
            product_name="Short Review Ski",
            domain="ski",
            attribute_scores={"stiffness": 9},
            review_passages=["Too short.", "Also very stiff and demanding ski on hardpack groomers yeah."],
        ),
    ]
    pairs = generate_contrastive_pairs(products, min_pairs_per_attribute=1, seed=42)
    positives = {p.positive for p in pairs}
    assert "Too short." not in positives


def test_generate_pairs_oversamples_when_needed():
    products = [
        ProductProfile(
            product_id="SKI-X",
            product_name="Lone Ski",
            domain="ski",
            attribute_scores={"stiffness": 9},
            review_passages=["This is a single long review passage about stiffness and rigidity."],
        ),
    ]
    pairs = generate_contrastive_pairs(products, min_pairs_per_attribute=10, seed=42)
    stiffness_templates = set(ANCHOR_TEMPLATES["stiffness"])
    stiff_pairs = [p for p in pairs if p.anchor in stiffness_templates]
    assert len(stiff_pairs) >= 10, "Should oversample to reach min_pairs_per_attribute"
```

Run:

```bash
cd /Users/harvey/Development/sports/recommend && python -m pytest implementations/design_11_finetuned_embed/tests/test_pairs.py -v
```

---

### Task 2: Trainer

**Files:**
- Create: `implementations/design_11_finetuned_embed/trainer.py`

- [ ] **Step 2.1: Create `trainer.py` with `fine_tune()` and `export_model()`**

```python
"""Fine-tuning wrapper around sentence-transformers for Design #11."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from .pairs import ContrastivePair, TrainingConfig

logger = logging.getLogger(__name__)


def fine_tune(
    pairs: list[ContrastivePair],
    config: TrainingConfig,
) -> SentenceTransformer:
    """Fine-tune MiniLM on contrastive pairs.

    Uses MultipleNegativesRankingLoss which treats other in-batch positives
    as negatives. With batch_size=16, each example gets 15 implicit negatives.
    """
    model = SentenceTransformer(config.base_model)

    split_idx = int(len(pairs) * (1 - config.validation_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_examples = [
        InputExample(texts=[p.anchor, p.positive]) for p in train_pairs
    ]

    train_loader = DataLoader(
        train_examples,
        batch_size=config.batch_size,
        shuffle=True,
    )

    loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = None
    if val_pairs:
        val_sentences_1 = [p.anchor for p in val_pairs]
        val_sentences_2 = [p.positive for p in val_pairs]
        val_scores = [1.0] * len(val_pairs)
        evaluator = EmbeddingSimilarityEvaluator(
            val_sentences_1, val_sentences_2, val_scores,
            name="attribute-val",
        )

    warmup_steps = int(len(train_loader) * config.epochs * config.warmup_fraction)

    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=config.epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=len(train_loader) if evaluator else 0,
        output_path=config.output_dir,
        optimizer_params={"lr": config.learning_rate},
        show_progress_bar=False,
    )

    return model


def export_model(model: SentenceTransformer, config: TrainingConfig) -> Path:
    """Save the fine-tuned model for inference."""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model.save(str(output / "model"))

    with open(output / "training_config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    return output
```

- [ ] **Step 2.2: Create `tests/test_trainer.py` (tiny synthetic data) and run**

This test uses 2 products, 2 attributes, 5 pairs. It verifies training completes and produces a model that can encode text.

```python
"""Tests for trainer.py — fine-tuning on tiny synthetic data."""

import shutil
import tempfile

import numpy as np

from implementations.design_11_finetuned_embed.pairs import (
    ContrastivePair,
    TrainingConfig,
)
from implementations.design_11_finetuned_embed.trainer import export_model, fine_tune


def _make_tiny_pairs() -> list[ContrastivePair]:
    """5 contrastive pairs: enough to verify training runs."""
    return [
        ContrastivePair(anchor="very stiff ski", positive="Incredibly rigid and demanding on hardpack."),
        ContrastivePair(anchor="ski with high stiffness", positive="The stiffest ski I have ever used."),
        ContrastivePair(anchor="playful and fun ski", positive="Super fun to throw around in the park."),
        ContrastivePair(anchor="lively and energetic ski", positive="Bouncy and lively, great for tricks."),
        ContrastivePair(anchor="very stiff ski", positive="Firm underfoot with zero give."),
    ]


def test_fine_tune_runs_and_produces_model():
    tmpdir = tempfile.mkdtemp(prefix="design11_test_")
    try:
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=2e-5,
            validation_split=0.2,
            output_dir=tmpdir,
            seed=42,
        )
        pairs = _make_tiny_pairs()
        model = fine_tune(pairs, config)

        # Model should be able to encode text
        vec = model.encode("stiff ski", normalize_embeddings=True)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        # L2-normalized vector should have norm ~1
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_export_model_saves_files():
    tmpdir = tempfile.mkdtemp(prefix="design11_test_export_")
    try:
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            output_dir=tmpdir,
        )
        pairs = _make_tiny_pairs()
        model = fine_tune(pairs, config)
        output_path = export_model(model, config)

        assert (output_path / "model").exists()
        assert (output_path / "training_config.json").exists()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

Run:

```bash
cd /Users/harvey/Development/sports/recommend && python -m pytest implementations/design_11_finetuned_embed/tests/test_trainer.py -v --timeout=120
```

---

### Task 3: Vector Construction

**Files:**
- Create: `implementations/design_11_finetuned_embed/vectors.py`
- Create: `implementations/design_11_finetuned_embed/tests/test_vectors.py`

- [ ] **Step 3.1: Create `vectors.py`**

```python
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
    """Build a single vector per product from review passage embeddings.

    Each product vector is the L2-normalized mean of its passage embeddings.
    Handles both benchmark data format (id/name/text) and protocol format
    (product_id/product_name/review_text).
    """
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
    """Build per-attribute centroids from high-scoring products.

    For each attribute, the centroid is the L2-normalized mean of product
    vectors where that attribute scores >= threshold.
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
```

- [ ] **Step 3.2: Create `tests/test_vectors.py` and run tests**

```python
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
    # First chunk should contain first 3 sentences
    assert "First sentence." in chunks[0]


def test_chunk_review_skips_short():
    text = "Hi. Ok. Sure."
    chunks = chunk_review(text, max_sentences=3)
    # Each chunk must have >= 5 words
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
    # 4 products: 3 high stiffness, 1 low
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

    # Should have centroids for both attributes
    attr_names = {c.attribute for c in centroids}
    assert "stiffness" in attr_names, "Should build stiffness centroid (3 high products)"

    for c in centroids:
        assert c.centroid.shape == (384,)
        # Should be L2-normalized
        assert abs(np.linalg.norm(c.centroid) - 1.0) < 1e-5


def test_build_attribute_centroids_skips_insufficient():
    rng = np.random.RandomState(42)
    # Only 1 product with high stiffness — need >= 2 for centroid
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
```

Run:

```bash
cd /Users/harvey/Development/sports/recommend && python -m pytest implementations/design_11_finetuned_embed/tests/test_vectors.py -v
```

---

### Task 4: Ranker

**Files:**
- Create: `implementations/design_11_finetuned_embed/ranker.py`
- Create: `implementations/design_11_finetuned_embed/tests/test_ranker.py`

- [ ] **Step 4.1: Create `ranker.py`**

```python
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
    """Rank products by cosine similarity to query in fine-tuned space.

    One encoder forward pass + one matrix multiply. No reranker, no LLM.
    """
    query_vec = model.encode(query_text, normalize_embeddings=True)

    product_matrix = np.stack([pv.vector for pv in product_vectors])

    # Cosine similarity (vectors are L2-normalized)
    scores = product_matrix @ query_vec

    # Min-max normalize to [0, 1]
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
    """Decompose query-product similarity into per-attribute contributions.

    For each attribute centroid C_a:
      contribution_a = (query . C_a) * (product . C_a)

    Returns:
        explanation: Human-readable string.
        matched_attributes: Dict of attribute -> contribution score (0-1 normalized).
    """
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
```

- [ ] **Step 4.2: Create `tests/test_ranker.py` and run tests**

```python
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

    # Create a query vector and product vector that are similar
    query_vec = _normed(rng.randn(dim).astype(np.float32))
    product_vec = _normed(query_vec + 0.1 * rng.randn(dim).astype(np.float32))

    # Create centroids aligned with query direction
    stiffness_centroid = _normed(query_vec + 0.05 * rng.randn(dim).astype(np.float32))
    # Orthogonal centroid — low contribution
    orth = rng.randn(dim).astype(np.float32)
    orth = orth - (orth @ query_vec) * query_vec  # remove query component
    playfulness_centroid = _normed(orth)

    centroids = [
        AttributeCentroid(attribute="stiffness", centroid=stiffness_centroid, high_scoring_ids=["A"]),
        AttributeCentroid(attribute="playfulness", centroid=playfulness_centroid, high_scoring_ids=["B"]),
    ]

    explanation, matched = explain_match(query_vec, product_vec, centroids)

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert isinstance(matched, dict)
    # Stiffness should dominate since its centroid is aligned with query
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

    # Build aligned vectors so contribution is > 0.01
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
    """Verify higher cosine sim product ranks first using a mock model."""
    dim = 384
    rng = np.random.RandomState(42)

    query_direction = _normed(rng.randn(dim).astype(np.float32))

    # Product A: very similar to query
    vec_a = _normed(query_direction + 0.05 * rng.randn(dim).astype(np.float32))
    # Product B: less similar
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

    # Use rank_products with a mock model
    class MockModel:
        def encode(self, text, normalize_embeddings=True, show_progress_bar=False):
            return query_direction

    from implementations.design_11_finetuned_embed.ranker import rank_products

    ranked = rank_products("stiff ski", [pv_a, pv_b], MockModel(), top_k=2)
    assert ranked[0][0].product_id == "A", "Product closer to query should rank first"
    assert ranked[0][1] >= ranked[1][1], "Scores should be descending"
```

Run:

```bash
cd /Users/harvey/Development/sports/recommend && python -m pytest implementations/design_11_finetuned_embed/tests/test_ranker.py -v
```

---

### Task 5: Recommender

**Files:**
- Create: `implementations/design_11_finetuned_embed/recommender.py`
- Create: `implementations/design_11_finetuned_embed/tests/test_recommender.py`

- [ ] **Step 5.1: Create `recommender.py`**

```python
"""Design #11: Fine-Tuned Embedding Ranker.

The embedding space IS the recommendation engine. Query-time cost is one
encoder forward pass + cosine similarity. No LLM calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the shared package is importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult  # noqa: E402

from .pairs import (  # noqa: E402
    ANCHOR_TEMPLATES,
    ProductProfile,
    TrainingConfig,
    generate_contrastive_pairs,
)
from .ranker import explain_match, rank_products  # noqa: E402
from .trainer import export_model, fine_tune  # noqa: E402
from .vectors import (  # noqa: E402
    AttributeCentroid,
    ProductVector,
    build_attribute_centroids,
    build_product_vectors,
    chunk_review,
)

logger = logging.getLogger(__name__)


class FineTunedEmbeddingRecommender:
    """Design #11: Fine-tuned embedding ranker.

    The embedding space IS the recommendation engine. Query-time cost is
    one encoder forward pass + cosine similarity. No LLM calls.
    """

    def __init__(self, model_dir: str | None = None):
        self.model = None  # SentenceTransformer, loaded lazily
        self.model_dir = model_dir
        self.product_vectors: dict[str, list[ProductVector]] = {}
        self.centroids: dict[str, list[AttributeCentroid]] = {}
        self._trained = False

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Ingest products and reviews: train (if needed) and build index."""
        from sentence_transformers import SentenceTransformer

        profiles = self._build_profiles(products, reviews, domain)

        if self.model is None:
            if self.model_dir and Path(self.model_dir, "model").exists():
                logger.info("Loading pre-trained model from %s", self.model_dir)
                self.model = SentenceTransformer(str(Path(self.model_dir) / "model"))
                self._trained = True
            else:
                logger.info("Training new model from %d profiles", len(profiles))
                self.model = self._train(profiles)
                self._trained = True

        pvecs = build_product_vectors(products, reviews, self.model)
        self.product_vectors[domain] = pvecs

        attributes = list(ANCHOR_TEMPLATES.keys())
        self.centroids[domain] = build_attribute_centroids(pvecs, attributes)
        logger.info(
            "Ingested %d products for domain '%s' (%d centroids)",
            len(pvecs), domain, len(self.centroids[domain]),
        )

    def query(
        self, query_text: str, domain: str, top_k: int = 10,
    ) -> list[RecommendationResult]:
        """Query the fine-tuned embedding space."""
        if domain not in self.product_vectors or self.model is None:
            return []

        pvecs = self.product_vectors[domain]
        centroids = self.centroids.get(domain, [])

        query_vec = self.model.encode(query_text, normalize_embeddings=True)

        ranked = rank_products(query_text, pvecs, self.model, top_k=top_k)

        results = []
        for pv, score in ranked:
            explanation, matched_attrs = explain_match(
                query_vec, pv.vector, centroids,
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
            pid = r.get("product_id", r.get("id", ""))
            text = r.get("review_text", r.get("text", ""))
            if pid and text:
                if pid not in reviews_by_product:
                    reviews_by_product[pid] = []
                reviews_by_product[pid].append(text)

        profiles = []
        for p in products:
            pid = p.get("product_id", p.get("id", ""))
            pname = p.get("product_name", p.get("name", ""))

            passages = []
            for text in reviews_by_product.get(pid, []):
                passages.extend(chunk_review(text))

            profiles.append(ProductProfile(
                product_id=pid,
                product_name=pname,
                domain=domain,
                attribute_scores=p.get("attributes", {}),
                review_passages=passages,
                metadata=p.get("specs", {}),
            ))
        return profiles

    def _train(self, profiles: list[ProductProfile]):
        """Train the fine-tuned model from product profiles."""
        config = TrainingConfig()
        if self.model_dir:
            config.output_dir = self.model_dir

        pairs = generate_contrastive_pairs(profiles)
        logger.info("Generated %d contrastive pairs", len(pairs))

        model = fine_tune(pairs, config)
        export_model(model, config)
        return model
```

- [ ] **Step 5.2: Create `tests/test_recommender.py` (integration test) and run**

This test uses a small slice of synthetic data to verify the full pipeline.

```python
"""Integration test for the FineTunedEmbeddingRecommender."""

import shutil
import tempfile

from shared.interface import RecommendationResult, Recommender

from implementations.design_11_finetuned_embed.recommender import (
    FineTunedEmbeddingRecommender,
)


def _make_small_dataset() -> tuple[list[dict], list[dict]]:
    """Small synthetic dataset: 3 products, multiple reviews each."""
    products = [
        {
            "id": "SKI-001",
            "name": "Race Carver",
            "brand": "TestBrand",
            "category": "race",
            "specs": {"waist_width_mm": 66},
            "attributes": {
                "stiffness": 9,
                "damp": 8,
                "edge_grip": 10,
                "stability_at_speed": 9,
                "playfulness": 2,
                "powder_float": 1,
                "forgiveness": 2,
            },
        },
        {
            "id": "SKI-002",
            "name": "Park Playful",
            "brand": "TestBrand",
            "category": "park",
            "specs": {"waist_width_mm": 90},
            "attributes": {
                "stiffness": 3,
                "damp": 3,
                "edge_grip": 4,
                "stability_at_speed": 3,
                "playfulness": 9,
                "powder_float": 5,
                "forgiveness": 8,
            },
        },
        {
            "id": "SKI-003",
            "name": "All Mountain",
            "brand": "TestBrand",
            "category": "all_mountain",
            "specs": {"waist_width_mm": 100},
            "attributes": {
                "stiffness": 6,
                "damp": 6,
                "edge_grip": 6,
                "stability_at_speed": 6,
                "playfulness": 6,
                "powder_float": 7,
                "forgiveness": 6,
            },
        },
    ]
    reviews = [
        {"product_id": "SKI-001", "text": "Absolutely the stiffest ski I have ever used. Incredible edge grip on hardpack, feels like rails. Very damp and composed even at high speed."},
        {"product_id": "SKI-001", "text": "Race-level performance with incredible stability. Locked in on ice and groomers. Not forgiving at all if you get lazy."},
        {"product_id": "SKI-001", "text": "The dampest race ski available. Zero chatter at any speed. Demands strong technique but rewards aggressive skiing."},
        {"product_id": "SKI-002", "text": "Super playful and fun in the park. Easy to butter and spin. Soft flex makes it very forgiving when you land backseat."},
        {"product_id": "SKI-002", "text": "Lively and energetic ski that pops off every feature. Light and easy to throw around. Not stable at speed though."},
        {"product_id": "SKI-002", "text": "Best park ski I have tried. Very forgiving soft flex, perfect for learning tricks. No edge grip on ice but who cares in the park."},
        {"product_id": "SKI-003", "text": "Does everything reasonably well. Decent on groomers, handles some powder, okay in bumps. A true quiver of one ski."},
        {"product_id": "SKI-003", "text": "Versatile all mountain ski that works in most conditions. Floats surprisingly well in powder for a 100mm waist."},
        {"product_id": "SKI-003", "text": "Jack of all trades. Handles everything from groomers to light powder. Not the best at anything but competent everywhere."},
    ]
    return products, reviews


def test_recommender_satisfies_protocol():
    rec = FineTunedEmbeddingRecommender()
    assert isinstance(rec, Recommender)


def test_recommender_ingest_and_query():
    tmpdir = tempfile.mkdtemp(prefix="design11_integration_")
    try:
        products, reviews = _make_small_dataset()
        rec = FineTunedEmbeddingRecommender(model_dir=tmpdir)

        rec.ingest(products=products, reviews=reviews, domain="ski")

        results = rec.query("stiff damp carving ski for hardpack", domain="ski", top_k=3)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, RecommendationResult)
            assert 0.0 <= r.score <= 1.0
            assert isinstance(r.explanation, str)
            assert isinstance(r.matched_attributes, dict)
            assert r.product_id in {"SKI-001", "SKI-002", "SKI-003"}
            assert r.product_name != ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_recommender_query_unknown_domain_returns_empty():
    rec = FineTunedEmbeddingRecommender()
    results = rec.query("anything", domain="nonexistent")
    assert results == []
```

Run:

```bash
cd /Users/harvey/Development/sports/recommend && python -m pytest implementations/design_11_finetuned_embed/tests/test_recommender.py -v --timeout=300
```

---

### Task 6: Package Setup

**Files:**
- Create: `implementations/design_11_finetuned_embed/__init__.py`
- Create: `implementations/design_11_finetuned_embed/requirements.txt`

- [ ] **Step 6.1: Create `__init__.py` with factory**

```python
"""Design #11: Fine-Tuned Embedding Ranker."""

from .recommender import FineTunedEmbeddingRecommender


def create_recommender() -> FineTunedEmbeddingRecommender:
    """Factory function used by the benchmark runner to instantiate this design."""
    return FineTunedEmbeddingRecommender()
```

- [ ] **Step 6.2: Create `requirements.txt`**

```
sentence-transformers>=2.2.0
numpy>=1.24.0
torch>=2.0.0
pytest>=7.0.0
```

- [ ] **Step 6.3: Install dependencies and verify import**

```bash
cd /Users/harvey/Development/sports/recommend && pip install -r implementations/design_11_finetuned_embed/requirements.txt
```

Then verify the factory import works:

```bash
cd /Users/harvey/Development/sports/recommend && python -c "from implementations.design_11_finetuned_embed import create_recommender; rec = create_recommender(); print(f'Created: {type(rec).__name__}')"
```

---

### Task 7: Benchmark Smoke Test

**Files:** None created. This task runs the implementation against test queries.

- [ ] **Step 7.1: Run all unit tests**

```bash
cd /Users/harvey/Development/sports/recommend && python -m pytest implementations/design_11_finetuned_embed/tests/ -v --timeout=300
```

- [ ] **Step 7.2: Smoke test with 3 queries against benchmark data**

```bash
cd /Users/harvey/Development/sports/recommend && python -c "
import json
from pathlib import Path
from implementations.design_11_finetuned_embed import create_recommender
from shared.interface import RecommendationResult

data_dir = Path('benchmark/data')
with open(data_dir / 'ski_products.json') as f:
    ski_data = json.load(f)
with open(data_dir / 'ski_reviews.json') as f:
    ski_reviews = json.load(f)

rec = create_recommender()
rec.ingest(products=ski_data['products'], reviews=ski_reviews['reviews'], domain='ski')

queries = [
    'stiff damp carving ski for hardpack',
    'playful lightweight park ski',
    'versatile all-mountain ski for any condition',
]

for q in queries:
    print(f'\nQuery: {q}')
    results = rec.query(q, domain='ski', top_k=5)
    for r in results:
        assert isinstance(r, RecommendationResult)
        assert 0.0 <= r.score <= 1.0
        assert isinstance(r.matched_attributes, dict)
        assert isinstance(r.explanation, str)
        assert r.product_id != ''
        assert r.product_name != ''
        print(f'  {r.score:.3f} {r.product_name}: {r.explanation}')
    print(f'  OK — {len(results)} results, all conform to RecommendationResult schema')

print('\nSmoke test passed.')
"
```
