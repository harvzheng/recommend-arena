"""Integration test for the FineTunedEmbeddingRecommender."""

import shutil
import tempfile

from shared.interface import RecommendationResult, Recommender

from implementations.design_11_finetuned_embed.recommender import (
    FineTunedEmbeddingRecommender,
)


def _make_small_dataset() -> tuple[list[dict], list[dict]]:
    products = [
        {
            "id": "SKI-001",
            "name": "Race Carver",
            "brand": "TestBrand",
            "category": "race",
            "specs": {"waist_width_mm": 66},
            "attributes": {
                "stiffness": 9, "damp": 8, "edge_grip": 10,
                "stability_at_speed": 9, "playfulness": 2,
                "powder_float": 1, "forgiveness": 2,
            },
        },
        {
            "id": "SKI-002",
            "name": "Park Playful",
            "brand": "TestBrand",
            "category": "park",
            "specs": {"waist_width_mm": 90},
            "attributes": {
                "stiffness": 3, "damp": 3, "edge_grip": 4,
                "stability_at_speed": 3, "playfulness": 9,
                "powder_float": 5, "forgiveness": 8,
            },
        },
        {
            "id": "SKI-003",
            "name": "All Mountain",
            "brand": "TestBrand",
            "category": "all_mountain",
            "specs": {"waist_width_mm": 100},
            "attributes": {
                "stiffness": 6, "damp": 6, "edge_grip": 6,
                "stability_at_speed": 6, "playfulness": 6,
                "powder_float": 7, "forgiveness": 6,
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
