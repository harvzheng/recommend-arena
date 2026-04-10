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
    """A single training example: anchor text paired with a review passage."""
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


def generate_contrastive_pairs(
    products: list[ProductProfile],
    min_pairs_per_attribute: int = 50,
    seed: int = 42,
) -> list[ContrastivePair]:
    """Generate (anchor, positive) pairs from structured scores and review passages."""
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
