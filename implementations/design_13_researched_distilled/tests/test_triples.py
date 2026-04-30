"""Tests for the score-derived contrastive triple builder."""
from __future__ import annotations

from implementations.design_13_researched_distilled.teacher import TeacherJudgment
from implementations.design_13_researched_distilled.triples import (
    TrainingTriple,
    _ProductPassages,
    build_triples,
)


def _judgment(query_id: str, product_id: str, score: float) -> TeacherJudgment:
    return TeacherJudgment(
        query_id=query_id,
        product_id=product_id,
        score=score,
        matched_attributes={},
        explanation="",
    )


def test_build_triples_uses_absolute_thresholds() -> None:
    judgments = {
        "q1": [
            _judgment("q1", "p_high1", 0.9),
            _judgment("q1", "p_high2", 0.8),
            _judgment("q1", "p_low1", 0.2),
            _judgment("q1", "p_low2", 0.1),
            _judgment("q1", "p_mid", 0.5),
        ],
    }
    products = {
        "p_high1": _ProductPassages(["high passage 1"]),
        "p_high2": _ProductPassages(["high passage 2"]),
        "p_low1": _ProductPassages(["low passage 1"]),
        "p_low2": _ProductPassages(["low passage 2"]),
        "p_mid": _ProductPassages(["mid passage"]),
    }
    queries = {"q1": "stiff ski"}
    triples = build_triples(judgments, products, queries, seed=0)
    assert len(triples) == 2 * 2  # 2 positives × 2 negatives
    for t in triples:
        assert isinstance(t, TrainingTriple)
        assert t.query == "stiff ski"
        assert "high" in t.positive_passage
        assert "low" in t.negative_passage
        assert t.score_margin > 0.5


def test_build_triples_falls_back_to_relative_thresholds() -> None:
    # All scores in mid-band; no abs threshold separation possible.
    judgments = {
        "q1": [_judgment("q1", f"p{i}", 0.4 + i * 0.05) for i in range(8)],
    }
    products = {f"p{i}": _ProductPassages([f"passage {i}"]) for i in range(8)}
    queries = {"q1": "vague ski"}
    triples = build_triples(judgments, products, queries, seed=0)
    # Top quartile (n//4 = 2) × bottom quartile (2) = 4
    assert len(triples) == 4


def test_build_triples_skips_products_with_no_passages() -> None:
    judgments = {
        "q1": [
            _judgment("q1", "p_high1", 0.9),
            _judgment("q1", "p_high2", 0.85),
            _judgment("q1", "p_low1", 0.1),
            _judgment("q1", "p_low2", 0.05),
        ],
    }
    products = {
        "p_high1": _ProductPassages([]),  # no passages
        "p_high2": _ProductPassages(["good"]),
        "p_low1": _ProductPassages(["bad"]),
        "p_low2": _ProductPassages([]),
    }
    queries = {"q1": "test"}
    triples = build_triples(judgments, products, queries, seed=0)
    # Only 1 viable positive × 1 viable negative
    assert len(triples) == 1
    assert triples[0].positive_passage == "good"
    assert triples[0].negative_passage == "bad"
