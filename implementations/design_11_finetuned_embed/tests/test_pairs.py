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
    stiffness_templates = set(ANCHOR_TEMPLATES["stiffness"])
    assert anchors & stiffness_templates, "Expected stiffness anchors in output"
    play_templates = set(ANCHOR_TEMPLATES["playfulness"])
    assert anchors & play_templates, "Expected playfulness anchors in output"


def test_generate_pairs_positive_from_high_scoring():
    products = _make_products()
    pairs = generate_contrastive_pairs(products, min_pairs_per_attribute=5, seed=42)
    stiffness_templates = set(ANCHOR_TEMPLATES["stiffness"])
    stiff_pairs = [p for p in pairs if p.anchor in stiffness_templates]
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
