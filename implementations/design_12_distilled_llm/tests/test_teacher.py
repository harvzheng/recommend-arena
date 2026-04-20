"""Tests for teacher.py — prompt formatting, JSON parsing, caching."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.context import ProductContext
from implementations.design_12_distilled_llm.db import init_db
from implementations.design_12_distilled_llm.teacher import (
    TEACHER_PROMPT,
    TeacherJudgment,
    label_all_pairs,
    label_pair,
)


def _make_ctx(pid: str = "SKI-001", name: str = "Test Ski") -> ProductContext:
    return ProductContext(
        product_id=pid,
        product_name=name,
        domain="ski",
        context_text="Specs: Length: 170cm\n\nReview consensus (3 reviews): Great ski.",
        spec_summary="Length: 170cm",
        review_summary="Great ski.",
        review_count=3,
    )


def _mock_llm(response_dict: dict) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = json.dumps(response_dict)
    llm.llm_model = "mock-teacher"
    return llm


def test_teacher_prompt_contains_placeholders():
    formatted = TEACHER_PROMPT.format(
        domain="ski",
        query="stiff carving ski",
        product_name="Test Ski",
        product_context="Specs: Length: 170cm",
    )
    assert "stiff carving ski" in formatted
    assert "Test Ski" in formatted
    assert "Specs: Length: 170cm" in formatted
    assert "expert ski product recommender" in formatted


def test_label_pair_parses_teacher_response():
    response = {
        "score": 0.85,
        "explanation": "Strong match on stiffness and edge hold.",
        "matched_attributes": {"stiffness": 0.9, "edge_grip": 0.8},
    }
    llm = _mock_llm(response)
    ctx = _make_ctx()

    judgment = label_pair("stiff carving ski", ctx, llm, "ski")

    assert isinstance(judgment, TeacherJudgment)
    assert judgment.score == 0.85
    assert judgment.explanation == "Strong match on stiffness and edge hold."
    assert judgment.matched_attributes["stiffness"] == 0.9
    assert judgment.teacher_model == "mock-teacher"
    assert judgment.product_id == "SKI-001"


def test_label_pair_clamps_score():
    llm = _mock_llm({"score": 1.5, "explanation": "x", "matched_attributes": {}})
    judgment = label_pair("q", _make_ctx(), llm, "ski")
    assert judgment.score == 1.0

    llm2 = _mock_llm({"score": -0.2, "explanation": "x", "matched_attributes": {}})
    judgment2 = label_pair("q", _make_ctx(), llm2, "ski")
    assert judgment2.score == 0.0


def test_label_all_pairs_caches_and_skips():
    response = {
        "score": 0.7,
        "explanation": "Decent match.",
        "matched_attributes": {"flex": 0.6},
    }
    llm = _mock_llm(response)
    db = init_db(":memory:")
    ctx = _make_ctx()

    # First run: labels the pair
    results = label_all_pairs(["query1"], [ctx], llm, "ski", db)
    assert len(results) == 1
    assert llm.generate.call_count == 1

    # Second run: skips due to cache
    results2 = label_all_pairs(["query1"], [ctx], llm, "ski", db)
    assert len(results2) == 0
    assert llm.generate.call_count == 1  # no new calls

    db.close()


def test_label_all_pairs_multiple_queries_and_products():
    response = {
        "score": 0.5,
        "explanation": "Partial match.",
        "matched_attributes": {},
    }
    llm = _mock_llm(response)
    db = init_db(":memory:")
    products = [_make_ctx("SKI-001", "Ski A"), _make_ctx("SKI-002", "Ski B")]

    results = label_all_pairs(["q1", "q2"], products, llm, "ski", db)
    assert len(results) == 4  # 2 queries x 2 products
    assert llm.generate.call_count == 4

    db.close()
