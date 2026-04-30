"""Tests for design-13 SQLite schema + helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from implementations.design_13_researched_distilled.db import (
    SCHEMA_VERSION,
    init_schema,
    open_db,
)


def test_init_schema_creates_all_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "d13.db"
    conn = open_db(str(db_path))
    init_schema(conn)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cur.fetchall()]
    assert tables == [
        "products",
        "reviews",
        "schema_meta",
        "synthetic_queries",
        "teacher_judgments",
    ]


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "d13.db"
    conn = open_db(str(db_path))
    init_schema(conn)
    init_schema(conn)
    version = conn.execute(
        "SELECT value FROM schema_meta WHERE key='version'"
    ).fetchone()[0]
    assert version == str(SCHEMA_VERSION)


def test_judgments_unique_constraint(tmp_path: Path) -> None:
    db_path = tmp_path / "d13.db"
    conn = open_db(str(db_path))
    init_schema(conn)
    conn.execute(
        "INSERT INTO synthetic_queries(query_id, text, difficulty, "
        "seed_attributes_json, domain) VALUES (?, ?, ?, ?, ?)",
        ("q1", "stiff ski", "easy", "[]", "ski"),
    )
    conn.execute(
        "INSERT INTO products(product_id, product_name, domain, "
        "metadata_json, ingested_at) VALUES (?, ?, ?, ?, ?)",
        ("p1", "Test Ski", "ski", "{}", "2026-01-01"),
    )
    insert = (
        "INSERT INTO teacher_judgments(query_id, product_id, score, "
        "matched_attributes_json, explanation, evidence_json, teacher_model, "
        "research_calls, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    conn.execute(
        insert,
        ("q1", "p1", 0.8, "{}", "...", "[]", "claude", 1, "now"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            insert,
            ("q1", "p1", 0.9, "{}", "...", "[]", "claude", 1, "now"),
        )
