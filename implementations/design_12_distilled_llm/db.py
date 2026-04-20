"""SQLite schema and helpers for Design #12."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def init_db(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """Create or open the SQLite database and ensure all tables exist."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS product_contexts (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    context_text TEXT NOT NULL,
    spec_summary TEXT,
    review_summary TEXT,
    review_count INTEGER DEFAULT 0,
    metadata_json TEXT,
    built_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teacher_judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    product_id TEXT NOT NULL,
    score REAL NOT NULL,
    explanation TEXT NOT NULL,
    matched_attributes_json TEXT NOT NULL,
    teacher_model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(query, product_id, teacher_model)
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_model TEXT NOT NULL,
    lora_rank INTEGER NOT NULL,
    num_examples INTEGER NOT NULL,
    num_epochs INTEGER NOT NULL,
    final_loss REAL,
    adapter_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_judgments_query ON teacher_judgments(query);
CREATE INDEX IF NOT EXISTS idx_judgments_product ON teacher_judgments(product_id);
CREATE INDEX IF NOT EXISTS idx_contexts_domain ON product_contexts(domain);
"""


def upsert_product_context(
    conn: sqlite3.Connection,
    *,
    product_id: str,
    product_name: str,
    domain: str,
    context_text: str,
    spec_summary: str,
    review_summary: str,
    review_count: int,
    metadata: dict,
    built_at: str,
) -> None:
    """Insert or replace a product context row."""
    conn.execute(
        "INSERT OR REPLACE INTO product_contexts "
        "(product_id, product_name, domain, context_text, "
        "spec_summary, review_summary, review_count, metadata_json, built_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            product_id, product_name, domain, context_text,
            spec_summary, review_summary, review_count,
            json.dumps(metadata), built_at,
        ),
    )


def insert_teacher_judgment(
    conn: sqlite3.Connection,
    *,
    query: str,
    product_id: str,
    score: float,
    explanation: str,
    matched_attributes: dict[str, float],
    teacher_model: str,
    created_at: str,
) -> None:
    """Insert a teacher judgment, ignoring duplicates."""
    conn.execute(
        "INSERT OR IGNORE INTO teacher_judgments "
        "(query, product_id, score, explanation, "
        "matched_attributes_json, teacher_model, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            query, product_id, score, explanation,
            json.dumps(matched_attributes), teacher_model, created_at,
        ),
    )


def get_cached_judgment(
    conn: sqlite3.Connection,
    query: str,
    product_id: str,
) -> dict | None:
    """Return a cached teacher judgment or None."""
    row = conn.execute(
        "SELECT score, explanation, matched_attributes_json "
        "FROM teacher_judgments WHERE query = ? AND product_id = ?",
        (query, product_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "score": row["score"],
        "explanation": row["explanation"],
        "matched_attributes": json.loads(row["matched_attributes_json"]),
    }


def insert_training_run(
    conn: sqlite3.Connection,
    *,
    base_model: str,
    lora_rank: int,
    num_examples: int,
    num_epochs: int,
    final_loss: float | None,
    adapter_path: str,
    created_at: str,
) -> int:
    """Record a training run and return its id."""
    cur = conn.execute(
        "INSERT INTO training_runs "
        "(base_model, lora_rank, num_examples, num_epochs, "
        "final_loss, adapter_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            base_model, lora_rank, num_examples, num_epochs,
            final_loss, adapter_path, created_at,
        ),
    )
    return cur.lastrowid
