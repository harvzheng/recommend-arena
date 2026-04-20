"""Tests for inference.py — JSON extraction, parsing, fallback behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from implementations.design_12_distilled_llm.inference import (
    StudentJudgment,
    _extract_json,
    _parse_judgment,
)


class TestExtractJson:
    def test_clean_json(self):
        text = '{"score": 0.8, "explanation": "Good.", "matched_attributes": {}}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.8

    def test_markdown_fenced_json(self):
        text = '```json\n{"score": 0.7, "explanation": "OK.", "matched_attributes": {}}\n```'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.7

    def test_markdown_fenced_no_lang(self):
        text = '```\n{"score": 0.6, "explanation": "Fair.", "matched_attributes": {}}\n```'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.6

    def test_leading_text(self):
        text = 'Here is my evaluation:\n{"score": 0.5, "explanation": "Partial.", "matched_attributes": {}}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.5

    def test_nested_braces(self):
        text = '{"score": 0.9, "explanation": "Great.", "matched_attributes": {"stiffness": 0.9, "grip": 0.8}}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["matched_attributes"]["stiffness"] == 0.9

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError, match="No JSON object found"):
            _extract_json("No JSON here at all")

    def test_unterminated_json_raises(self):
        with pytest.raises(json.JSONDecodeError, match="Unterminated"):
            _extract_json('{"score": 0.5, "explanation": "cut off')

    def test_trailing_text(self):
        text = '{"score": 0.4, "explanation": "Low.", "matched_attributes": {}}\nSome extra text.'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["score"] == 0.4


class TestParseJudgment:
    def test_valid_output(self):
        raw = json.dumps({
            "score": 0.85,
            "explanation": "Strong match.",
            "matched_attributes": {"stiffness": 0.9},
        })
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.parse_success is True
        assert j.score == 0.85
        assert j.explanation == "Strong match."
        assert j.matched_attributes["stiffness"] == 0.9
        assert j.product_id == "SKI-001"

    def test_score_clamping(self):
        raw = json.dumps({"score": 2.0, "explanation": "x", "matched_attributes": {}})
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.score == 1.0

        raw2 = json.dumps({"score": -1.0, "explanation": "x", "matched_attributes": {}})
        j2 = _parse_judgment(raw2, "SKI-001", "Test Ski")
        assert j2.score == 0.0

    def test_malformed_json_fallback(self):
        j = _parse_judgment("not json at all", "SKI-001", "Test Ski")
        assert j.parse_success is False
        assert j.score == 0.0
        assert "Parse error" in j.explanation
        assert j.raw_output == "not json at all"

    def test_missing_score_key_fallback(self):
        raw = json.dumps({"explanation": "No score.", "matched_attributes": {}})
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.parse_success is False
        assert j.score == 0.0

    def test_missing_explanation_uses_empty(self):
        raw = json.dumps({"score": 0.5, "matched_attributes": {}})
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.parse_success is True
        assert j.explanation == ""

    def test_attribute_values_clamped(self):
        raw = json.dumps({
            "score": 0.5,
            "explanation": "x",
            "matched_attributes": {"a": 1.5, "b": -0.3},
        })
        j = _parse_judgment(raw, "SKI-001", "Test Ski")
        assert j.matched_attributes["a"] == 1.0
        assert j.matched_attributes["b"] == 0.0
