"""Tests for context.py — ProductContext construction with mocked LLM."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.context import (
    ProductContext,
    build_product_context,
    build_spec_summary,
)


def test_build_spec_summary_formats_metadata():
    metadata = {"length_cm": 177, "waist_width_mm": 100, "weight_g": 1850}
    result = build_spec_summary(metadata)
    assert "Length Cm: 177" in result
    assert "Waist Width Mm: 100" in result
    assert "Weight G: 1850" in result


def test_build_spec_summary_empty():
    assert build_spec_summary({}) == "No specs available."


def test_build_product_context_with_reviews():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Reviewers praise edge grip and stiffness."

    product = {
        "id": "SKI-001",
        "name": "Test Ski",
        "specs": {"length_cm": 170, "waist_width_mm": 66},
    }
    reviews = [
        {"product_id": "SKI-001", "text": "Great edge grip."},
        {"product_id": "SKI-001", "text": "Very stiff ski."},
    ]

    ctx = build_product_context(product, reviews, mock_llm, "ski")

    assert isinstance(ctx, ProductContext)
    assert ctx.product_id == "SKI-001"
    assert ctx.product_name == "Test Ski"
    assert ctx.domain == "ski"
    assert ctx.review_count == 2
    assert "Length Cm: 170" in ctx.spec_summary
    assert "Reviewers praise" in ctx.review_summary
    assert "Specs:" in ctx.context_text
    assert "Review consensus (2 reviews):" in ctx.context_text
    mock_llm.generate.assert_called_once()


def test_build_product_context_no_reviews():
    mock_llm = MagicMock()

    product = {"product_id": "SKI-002", "product_name": "Bare Ski", "metadata": {}}
    ctx = build_product_context(product, [], mock_llm, "ski")

    assert ctx.review_count == 0
    assert ctx.review_summary == "No reviews available."
    mock_llm.generate.assert_not_called()


def test_build_product_context_uses_alternate_keys():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Summary."

    product = {"product_id": "X-1", "product_name": "Alt Ski", "metadata": {"flex": 5}}
    reviews = [{"review_text": "Nice flex."}]

    ctx = build_product_context(product, reviews, mock_llm, "ski")
    assert ctx.product_id == "X-1"
    assert ctx.product_name == "Alt Ski"
