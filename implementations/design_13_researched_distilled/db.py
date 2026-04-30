"""SQLite schema and connection helpers for design #13.

One database per domain. Schema mirrors design-13 spec §3.4.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    review_text TEXT NOT NULL,
    source TEXT
);
CREATE TABLE IF NOT EXISTS synthetic_queries (
    query_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    seed_attributes_json TEXT NOT NULL,
    domain TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS teacher_judgments (
    query_id TEXT NOT NULL REFERENCES synthetic_queries(query_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    score REAL NOT NULL,
    matched_attributes_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    teacher_model TEXT NOT NULL,
    research_calls INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (query_id, product_id, teacher_model)
);
CREATE INDEX IF NOT EXISTS idx_judgments_query ON teacher_judgments(query_id);
CREATE INDEX IF NOT EXISTS idx_judgments_score ON teacher_judgments(score);
"""


def open_db(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
