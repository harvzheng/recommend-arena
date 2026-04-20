"""Tests for db.py — schema creation and helper functions."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from implementations.design_12_distilled_llm.db import (
    get_cached_judgment,
    init_db,
    insert_teacher_judgment,
    insert_training_run,
    upsert_product_context,
)


def test_init_db_creates_tables():
    conn = init_db(":memory:")
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "product_contexts" in tables
    assert "teacher_judgments" in tables
    assert "training_runs" in tables
    conn.close()


def test_upsert_product_context():
    conn = init_db(":memory:")
    upsert_product_context(
        conn,
        product_id="SKI-001",
        product_name="Test Ski",
        domain="ski",
        context_text="Specs: Length 170cm\n\nReviews: Great ski.",
        spec_summary="Length 170cm",
        review_summary="Great ski.",
        review_count=3,
        metadata={"length_cm": 170},
        built_at="2026-01-01T00:00:00",
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM product_contexts WHERE product_id = 'SKI-001'"
    ).fetchone()
    assert row is not None
    assert row["product_name"] == "Test Ski"
    assert row["review_count"] == 3
    conn.close()


def test_insert_and_cache_teacher_judgment():
    conn = init_db(":memory:")
    insert_teacher_judgment(
        conn,
        query="stiff carving ski",
        product_id="SKI-001",
        score=0.85,
        explanation="Strong match on stiffness.",
        matched_attributes={"stiffness": 0.9, "edge_grip": 0.8},
        teacher_model="claude-sonnet-4-20250514",
        created_at="2026-01-01T00:00:00",
    )
    conn.commit()

    cached = get_cached_judgment(conn, "stiff carving ski", "SKI-001")
    assert cached is not None
    assert cached["score"] == 0.85
    assert cached["matched_attributes"]["stiffness"] == 0.9

    miss = get_cached_judgment(conn, "powder ski", "SKI-001")
    assert miss is None
    conn.close()


def test_insert_training_run():
    conn = init_db(":memory:")
    run_id = insert_training_run(
        conn,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        lora_rank=16,
        num_examples=500,
        num_epochs=3,
        final_loss=0.42,
        adapter_path="adapters/design-12",
        created_at="2026-01-01T00:00:00",
    )
    conn.commit()
    assert run_id is not None
    row = conn.execute(
        "SELECT * FROM training_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert row["final_loss"] == 0.42
    conn.close()
