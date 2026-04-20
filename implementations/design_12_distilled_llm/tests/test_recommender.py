"""Integration tests for recommender.py with mocked student inference."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.context import ProductContext
from implementations.design_12_distilled_llm.inference import (
    StudentInference,
    StudentJudgment,
)
from implementations.design_12_distilled_llm.recommender import (
    DistilledLLMRecommender,
)
from shared.interface import RecommendationResult


def _make_mock_inference() -> MagicMock:
    mock = MagicMock(spec=StudentInference)

    def fake_infer(query: str, ctx: ProductContext) -> StudentJudgment:
        score = 0.8 if "carving" in ctx.product_name.lower() else 0.3
        return StudentJudgment(
            product_id=ctx.product_id,
            product_name=ctx.product_name,
            score=score,
            explanation=f"Evaluated {ctx.product_name} for: {query}",
            matched_attributes={"stiffness": score},
            raw_output="mock",
            parse_success=True,
        )

    mock.infer.side_effect = fake_infer
    return mock


def _make_mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = "Reviewers praise this ski."
    llm.llm_model = "mock-teacher"
    return llm


def test_recommender_ingest_and_query():
    mock_llm = _make_mock_llm()
    mock_inference = _make_mock_inference()

    rec = DistilledLLMRecommender(
        db_path=":memory:",
        llm=mock_llm,
        inference_backend=mock_inference,
    )

    products = [
        {"id": "SKI-001", "name": "Expert Carving Ski", "specs": {"length_cm": 170}},
        {"id": "SKI-002", "name": "Powder Floater", "specs": {"length_cm": 185}},
    ]
    reviews = [
        {"product_id": "SKI-001", "text": "Great edge grip."},
        {"product_id": "SKI-002", "text": "Floats in powder."},
    ]

    rec.ingest(products, reviews, "ski")

    assert len(rec.contexts) == 2
    assert "SKI-001" in rec.contexts
    assert "SKI-002" in rec.contexts

    results = rec.query("stiff carving ski", "ski", top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RecommendationResult) for r in results)
    assert results[0].product_id == "SKI-001"
    assert results[0].score > results[1].score
    assert 0.0 <= results[0].score <= 1.0
    assert len(results[0].explanation) > 0
    assert isinstance(results[0].matched_attributes, dict)


def test_recommender_query_empty_domain():
    mock_llm = _make_mock_llm()
    mock_inference = _make_mock_inference()

    rec = DistilledLLMRecommender(
        db_path=":memory:",
        llm=mock_llm,
        inference_backend=mock_inference,
    )

    results = rec.query("anything", "ski", top_k=5)
    assert results == []


def test_recommender_top_k_limits_results():
    mock_llm = _make_mock_llm()
    mock_inference = _make_mock_inference()

    rec = DistilledLLMRecommender(
        db_path=":memory:",
        llm=mock_llm,
        inference_backend=mock_inference,
    )

    products = [
        {"id": f"SKI-{i:03d}", "name": f"Ski {i}", "specs": {}}
        for i in range(10)
    ]
    rec.ingest(products, [], "ski")

    results = rec.query("test", "ski", top_k=3)
    assert len(results) == 3


def test_recommender_stores_contexts_in_db():
    mock_llm = _make_mock_llm()

    rec = DistilledLLMRecommender(
        db_path=":memory:",
        llm=mock_llm,
        inference_backend=_make_mock_inference(),
    )

    products = [{"id": "SKI-001", "name": "Test Ski", "specs": {"flex": 7}}]
    rec.ingest(products, [], "ski")

    row = rec.db.execute(
        "SELECT * FROM product_contexts WHERE product_id = 'SKI-001'"
    ).fetchone()
    assert row is not None
    assert row["product_name"] == "Test Ski"
    assert row["domain"] == "ski"
