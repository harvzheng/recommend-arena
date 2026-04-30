"""Tests for the synthetic query generator (Phase 1)."""
from __future__ import annotations

import json

from implementations.design_13_researched_distilled.synthetic_queries import (
    SyntheticQuery,
    _normalize_query,
    generate_synthetic_queries,
)


class FakeLLM:
    """Returns canned JSON responses, records calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, **kw: object) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def test_generate_returns_balanced_buckets() -> None:
    canned: list[str] = []
    canned += [
        json.dumps({"queries": [{"text": f"easy q {i}", "rationale": "x"} for i in range(5)]})
    ] * 8
    canned += [
        json.dumps({"queries": [{"text": f"medium q {i}", "rationale": "x"} for i in range(5)]})
    ] * 16
    canned += [
        json.dumps({"queries": [{"text": f"hard q {i}", "rationale": "x"} for i in range(5)]})
    ] * 12
    canned += [
        json.dumps({"queries": [{"text": f"vague q {i}", "rationale": "x"} for i in range(5)]})
    ] * 10
    canned += [
        json.dumps({"queries": [{"text": f"cross q {i}", "rationale": "x"} for i in range(3)]})
    ] * 5

    llm = FakeLLM(canned)
    queries = generate_synthetic_queries(
        domain="ski",
        attributes=[
            "stiffness", "edge_grip", "dampness", "stability",
            "playfulness", "weight", "turn_initiation", "versatility",
        ],
        llm=llm,
        seed=42,
    )

    by_bucket: dict[str, int] = {}
    for q in queries:
        by_bucket[q.difficulty] = by_bucket.get(q.difficulty, 0) + 1

    assert by_bucket == {
        "easy": 40, "medium": 80, "hard": 60, "vague": 50, "cross_domain": 15,
    }
    assert all(isinstance(q, SyntheticQuery) for q in queries)
    assert all(q.query_id.startswith("syn-") for q in queries)


def test_generate_excludes_benchmark_queries() -> None:
    canned = [
        json.dumps({"queries": [
            {"text": "powder ski with good float", "rationale": "x"},
            {"text": "stiff carving ski", "rationale": "x"},
        ]})
    ]
    llm = FakeLLM(canned)
    benchmark = {"powder ski with good float"}
    out = generate_synthetic_queries(
        domain="ski",
        attributes=["stiffness"],
        llm=llm,
        seed=0,
        benchmark_queries=benchmark,
        max_combos_per_bucket=1,
        buckets=("easy",),
    )
    texts = [q.text for q in out]
    assert "powder ski with good float" not in texts
    assert "stiff carving ski" in texts


def test_normalize_query_dedupe() -> None:
    assert _normalize_query("Stiff Carving Ski.") == _normalize_query("stiff carving ski")
