"""Tests for the design-13 teacher labeling pipeline."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from implementations.design_13_researched_distilled.db import init_schema, open_db
from implementations.design_13_researched_distilled.synthetic_queries import (
    SyntheticQuery,
)
from implementations.design_13_researched_distilled.teacher import (
    TeacherJudgment,
    label_all_pairs,
    label_pair,
)


class _FakeResp:
    def __init__(self, text: str, tool_calls: int) -> None:
        self.text = text
        self.tool_calls = tool_calls


class FakeResearchedLLM:
    def __init__(self, responses: list[tuple[str, int]]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []
        self.model = "fake-claude"

    def research_generate(self, prompt: str, **_: object) -> _FakeResp:
        self.calls.append(prompt)
        text, n_calls = self._responses.pop(0)
        return _FakeResp(text=text, tool_calls=n_calls)


def _seed(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO products(product_id, product_name, domain, "
        "metadata_json, ingested_at) VALUES (?, ?, ?, ?, ?)",
        ("p1", "Ski A", "ski", "{}", "2026-01-01"),
    )
    conn.execute(
        "INSERT INTO synthetic_queries(query_id, text, difficulty, "
        "seed_attributes_json, domain) VALUES (?, ?, ?, ?, ?)",
        ("q1", "stiff ski", "easy", "[]", "ski"),
    )
    conn.commit()


def test_label_pair_parses_and_clamps() -> None:
    raw = json.dumps({
        "score": 1.5,  # gets clamped
        "matched_attributes": {"stiffness": 0.9},
        "explanation": "stiff and good",
        "evidence": [{"source": "r1", "text": "stiff", "relevance": 0.9}],
    })
    llm = FakeResearchedLLM([(raw, 1)])
    j = label_pair(
        query=SyntheticQuery(
            query_id="q1", text="stiff ski", difficulty="easy", domain="ski",
        ),
        product_name="Ski A",
        product_id="p1",
        product_context="A stiff ski.",
        domain="ski",
        llm=llm,
    )
    assert isinstance(j, TeacherJudgment)
    assert j.score == 1.0
    assert j.matched_attributes == {"stiffness": 0.9}
    assert j.research_calls == 1
    assert j.teacher_model == "fake-claude"


def test_label_all_pairs_caches(tmp_path: Path) -> None:
    db_path = tmp_path / "d.db"
    conn = open_db(str(db_path))
    init_schema(conn)
    _seed(conn)

    raw = json.dumps({
        "score": 0.7,
        "matched_attributes": {},
        "explanation": "x",
        "evidence": [],
    })
    llm = FakeResearchedLLM([(raw, 0)])
    queries = [SyntheticQuery(
        query_id="q1", text="stiff ski", difficulty="easy", domain="ski",
    )]
    products = [{
        "product_id": "p1",
        "product_name": "Ski A",
        "context_text": "A stiff ski.",
    }]

    label_all_pairs(queries, products, "ski", llm, conn)
    label_all_pairs(queries, products, "ski", llm, conn)  # second run, cache

    assert len(llm.calls) == 1
    rows = conn.execute("SELECT COUNT(*) FROM teacher_judgments").fetchone()[0]
    assert rows == 1


def test_label_pair_rejects_invalid_json() -> None:
    llm = FakeResearchedLLM([("not json", 0)])
    with pytest.raises(ValueError):
        label_pair(
            query=SyntheticQuery(
                query_id="q1", text="x", difficulty="easy", domain="ski",
            ),
            product_name="A",
            product_id="p1",
            product_context="ctx",
            domain="ski",
            llm=llm,
        )
