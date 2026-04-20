"""SQLite structured storage for products, reviews, and aspects."""

from __future__ import annotations

import json
import logging
import sqlite3

logger = logging.getLogger(__name__)


class StructuredStore:
    """SQLite-backed store for products, reviews, and extracted aspects."""

    def __init__(self, db_path: str = ":memory:"):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS products (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                domain      TEXT NOT NULL,
                raw_meta    TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS reviews (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id  TEXT REFERENCES products(id),
                source      TEXT,
                author      TEXT,
                text        TEXT NOT NULL,
                overall_sentiment REAL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS aspects (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id   INTEGER REFERENCES reviews(id),
                product_id  TEXT REFERENCES products(id),
                raw_aspect  TEXT NOT NULL,
                norm_aspect TEXT NOT NULL,
                opinion     TEXT,
                sentiment   REAL NOT NULL,
                confidence  REAL DEFAULT 1.0
            );

            CREATE INDEX IF NOT EXISTS idx_aspects_product ON aspects(product_id);
            CREATE INDEX IF NOT EXISTS idx_aspects_norm ON aspects(norm_aspect);
            CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
            """
        )
        # Create the view — drop first to allow re-creation on re-ingest
        self.db.execute("DROP VIEW IF EXISTS product_attributes")
        self.db.execute(
            """
            CREATE VIEW product_attributes AS
            SELECT
                product_id,
                norm_aspect,
                AVG(sentiment) AS avg_sentiment,
                COUNT(*) AS mention_count
            FROM aspects
            GROUP BY product_id, norm_aspect
            """
        )
        self.db.commit()

    def upsert_product(
        self,
        product_id: str,
        name: str,
        domain: str,
        metadata: dict | None = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO products (id, name, domain, raw_meta)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                domain = excluded.domain,
                raw_meta = excluded.raw_meta
            """,
            (product_id, name, domain, json.dumps(metadata or {})),
        )
        self.db.commit()

    def insert_review(
        self,
        product_id: str,
        text: str,
        author: str | None = None,
        source: str | None = None,
        overall_sentiment: float | None = None,
    ) -> int:
        cursor = self.db.execute(
            """
            INSERT INTO reviews (product_id, text, author, source, overall_sentiment)
            VALUES (?, ?, ?, ?, ?)
            """,
            (product_id, text, author, source, overall_sentiment),
        )
        self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def insert_aspects(
        self,
        review_id: int,
        product_id: str,
        aspects: list[dict],
    ) -> None:
        rows = [
            (
                review_id,
                product_id,
                a["raw_aspect"],
                a["norm_aspect"],
                a.get("opinion", ""),
                a["sentiment"],
                a.get("confidence", 1.0),
            )
            for a in aspects
        ]
        self.db.executemany(
            """
            INSERT INTO aspects (review_id, product_id, raw_aspect, norm_aspect, opinion, sentiment, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.db.commit()

    def get_product_aspects(self, product_id: str) -> dict[str, float]:
        """Return {norm_aspect: avg_sentiment} for a product."""
        rows = self.db.execute(
            "SELECT norm_aspect, avg_sentiment FROM product_attributes WHERE product_id = ?",
            (product_id,),
        ).fetchall()
        return {row["norm_aspect"]: row["avg_sentiment"] for row in rows}

    def get_product_name(self, product_id: str) -> str:
        row = self.db.execute(
            "SELECT name FROM products WHERE id = ?", (product_id,)
        ).fetchone()
        return row["name"] if row else product_id

    def get_product_summary_data(self, product_id: str) -> list[tuple[str, float, int]]:
        """Return [(norm_aspect, avg_sentiment, mention_count), ...] ordered by mention count."""
        rows = self.db.execute(
            """
            SELECT norm_aspect, AVG(sentiment) as avg_sent, COUNT(*) as cnt
            FROM aspects
            WHERE product_id = ?
            GROUP BY norm_aspect
            ORDER BY cnt DESC
            """,
            (product_id,),
        ).fetchall()
        return [(row["norm_aspect"], row["avg_sent"], row["cnt"]) for row in rows]

    def get_all_product_ids(self, domain: str | None = None) -> list[str]:
        if domain:
            rows = self.db.execute(
                "SELECT id FROM products WHERE domain = ?", (domain,)
            ).fetchall()
        else:
            rows = self.db.execute("SELECT id FROM products").fetchall()
        return [row["id"] for row in rows]

    def filter_by_negations(
        self,
        candidate_ids: list[str],
        negations: list[str],
    ) -> list[str]:
        """Remove candidates that have positively-reviewed negated aspects."""
        if not negations or not candidate_ids:
            return candidate_ids

        placeholders = ",".join("?" * len(candidate_ids))
        query = (
            f"SELECT DISTINCT product_id FROM product_attributes "
            f"WHERE product_id IN ({placeholders})"
        )
        params: list = list(candidate_ids)

        for neg in negations:
            query += (
                " AND product_id NOT IN ("
                "SELECT product_id FROM aspects WHERE norm_aspect = ? AND sentiment > 0"
                ")"
            )
            params.append(neg)

        rows = self.db.execute(query, params).fetchall()
        return [row["product_id"] for row in rows]

    def clear_domain(self, domain: str) -> None:
        """Remove all data for a domain to support re-ingestion."""
        product_ids = self.get_all_product_ids(domain)
        if not product_ids:
            return
        placeholders = ",".join("?" * len(product_ids))
        self.db.execute(
            f"DELETE FROM aspects WHERE product_id IN ({placeholders})", product_ids
        )
        self.db.execute(
            f"DELETE FROM reviews WHERE product_id IN ({placeholders})", product_ids
        )
        self.db.execute(
            f"DELETE FROM products WHERE id IN ({placeholders})", product_ids
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
