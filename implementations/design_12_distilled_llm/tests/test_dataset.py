"""Tests for dataset.py — training data format and JSONL output."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.context import ProductContext
from implementations.design_12_distilled_llm.dataset import (
    STUDENT_INSTRUCTION,
    TrainingExample,
    build_training_dataset,
    save_dataset_jsonl,
)
from implementations.design_12_distilled_llm.teacher import TeacherJudgment


def _make_judgment() -> TeacherJudgment:
    return TeacherJudgment(
        query="stiff carving ski",
        product_id="SKI-001",
        product_name="Test Ski",
        score=0.85,
        explanation="Strong match on stiffness.",
        matched_attributes={"stiffness": 0.9, "edge_grip": 0.8},
        teacher_model="mock-teacher",
        timestamp="2026-01-01T00:00:00",
    )


def _make_context() -> ProductContext:
    return ProductContext(
        product_id="SKI-001",
        product_name="Test Ski",
        domain="ski",
        context_text="Specs: Length: 170cm\n\nReview consensus (3 reviews): Great ski.",
        spec_summary="Length: 170cm",
        review_summary="Great ski.",
        review_count=3,
    )


def test_build_training_dataset_format():
    judgments = [_make_judgment()]
    contexts = {"SKI-001": _make_context()}

    dataset = build_training_dataset(judgments, contexts)

    assert len(dataset) == 1
    ex = dataset[0]
    assert isinstance(ex, TrainingExample)
    assert ex.instruction == STUDENT_INSTRUCTION
    assert "stiff carving ski" in ex.input
    assert "Test Ski" in ex.input
    assert "Specs: Length: 170cm" in ex.input

    output_parsed = json.loads(ex.output)
    assert output_parsed["score"] == 0.85
    assert output_parsed["explanation"] == "Strong match on stiffness."
    assert output_parsed["matched_attributes"]["stiffness"] == 0.9


def test_training_example_to_dict():
    ex = TrainingExample(
        instruction="Inst",
        input="In",
        output="Out",
    )
    d = ex.to_dict()
    assert d == {"instruction": "Inst", "input": "In", "output": "Out"}


def test_save_dataset_jsonl():
    judgments = [_make_judgment()]
    contexts = {"SKI-001": _make_context()}
    dataset = build_training_dataset(judgments, contexts)

    with tempfile.TemporaryDirectory() as tmp:
        path = save_dataset_jsonl(dataset, Path(tmp) / "train.jsonl")
        assert path.exists()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert "instruction" in record
        assert "input" in record
        assert "output" in record
        assert "stiff carving ski" in record["input"]


def test_save_dataset_jsonl_multiple_examples():
    j1 = _make_judgment()
    j2 = TeacherJudgment(
        query="powder ski",
        product_id="SKI-001",
        product_name="Test Ski",
        score=0.3,
        explanation="Poor match for powder.",
        matched_attributes={"powder_float": 0.1},
        teacher_model="mock-teacher",
        timestamp="2026-01-01T00:00:00",
    )
    contexts = {"SKI-001": _make_context()}
    dataset = build_training_dataset([j1, j2], contexts)

    with tempfile.TemporaryDirectory() as tmp:
        path = save_dataset_jsonl(dataset, Path(tmp) / "train.jsonl")
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert "stiff carving ski" in r1["input"]
        assert "powder ski" in r2["input"]
