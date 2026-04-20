"""Index wrappers for BM25 (rank_bm25), FAISS flat, and SQLite.

Each index is built during ingestion and queried at recommendation time.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 Index (in-memory via rank_bm25)
# ---------------------------------------------------------------------------

class BM25Index:
    """In-memory BM25 index over per-product review corpora."""

    def __init__(self) -> None:
        self.product_ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build(self, product_docs: dict[str, str]) -> None:
        """Build index from product_id -> concatenated review text mapping."""
        self.product_ids = list(product_docs.keys())
        tokenized = [
            product_docs[pid].lower().split() for pid in self.product_ids
        ]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d documents", len(self.product_ids))

    def score_all(self, query_text: str) -> dict[str, float]:
        """Return BM25 scores for all indexed products."""
        if self.bm25 is None:
            return {}
        tokens = query_text.lower().split()
        scores = self.bm25.get_scores(tokens)
        return dict(zip(self.product_ids, scores.tolist()))

    def pid_list(self) -> list[str]:
        return list(self.product_ids)


# ---------------------------------------------------------------------------
# FAISS Flat Index
# ---------------------------------------------------------------------------

class FAISSIndex:
    """FAISS flat (brute-force) index over product embeddings."""

    def __init__(self) -> None:
        self.product_ids: list[str] = []
        self.index = None  # faiss.IndexFlatIP
        self.dim: int = 0

    def build(self, product_embeddings: dict[str, list[float]]) -> None:
        """Build FAISS flat index from product_id -> embedding mapping."""
        import faiss

        self.product_ids = list(product_embeddings.keys())
        if not self.product_ids:
            return

        vecs = np.array(
            [product_embeddings[pid] for pid in self.product_ids],
            dtype=np.float32,
        )
        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms

        self.dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        logger.info(
            "FAISS index built with %d vectors of dim %d",
            len(self.product_ids),
            self.dim,
        )

    def score_all(self, query_embedding: list[float]) -> dict[str, float]:
        """Return cosine similarity scores for all indexed products."""
        if self.index is None or not self.product_ids:
            return {}

        q = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        k = len(self.product_ids)
        distances, indices = self.index.search(q, k)

        result = {}
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.product_ids):
                result[self.product_ids[idx]] = float(dist)
        return result


# ---------------------------------------------------------------------------
# SQLite Structured Store
# ---------------------------------------------------------------------------

class StructuredStore:
    """SQLite-backed structured product store with attribute scores."""

    def __init__(self) -> None:
        self.conn: sqlite3.Connection | None = None

    def build(
        self,
        products: list[dict],
        extracted_attributes: dict[str, list[dict]],
        domain: str,
    ) -> None:
        """Build SQLite database from products and extracted attributes.

        Args:
            products: Raw product dicts from benchmark data.
            extracted_attributes: product_id -> list of
                {aspect, sentiment, confidence} dicts.
            domain: Domain identifier.
        """
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()

        cur.executescript("""
            CREATE TABLE products (
                id          TEXT PRIMARY KEY,
                domain      TEXT NOT NULL,
                name        TEXT NOT NULL,
                brand       TEXT,
                category    TEXT,
                metadata    TEXT
            );

            CREATE TABLE product_attribute_scores (
                product_id  TEXT,
                aspect      TEXT,
                avg_sentiment   REAL,
                mention_count   INTEGER,
                confidence      REAL,
                PRIMARY KEY (product_id, aspect)
            );

            CREATE TABLE product_reviews (
                product_id TEXT,
                review_count INTEGER,
                avg_rating REAL,
                PRIMARY KEY (product_id)
            );
        """)

        # Insert products
        for p in products:
            pid = p.get("product_id", p.get("id", ""))
            cur.execute(
                "INSERT OR REPLACE INTO products (id, domain, name, brand, category, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    pid,
                    domain,
                    p.get("product_name", p.get("name", "")),
                    p.get("brand", ""),
                    p.get("category", ""),
                    json.dumps(p.get("metadata", p.get("specs", {}))),
                ),
            )

        # Insert attribute scores (already aggregated per product)
        for pid, attrs in extracted_attributes.items():
            for attr in attrs:
                cur.execute(
                    "INSERT OR REPLACE INTO product_attribute_scores "
                    "(product_id, aspect, avg_sentiment, mention_count, confidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        pid,
                        attr["aspect"],
                        attr.get("avg_sentiment", attr.get("sentiment", 0.0)),
                        attr.get("mention_count", 1),
                        attr.get("confidence", 1.0),
                    ),
                )

        self.conn.commit()
        logger.info("SQLite store built with %d products", len(products))

    def set_review_stats(
        self, review_counts: dict[str, int], avg_ratings: dict[str, float]
    ) -> None:
        """Store per-product review statistics."""
        if self.conn is None:
            return
        cur = self.conn.cursor()
        for pid in review_counts:
            cur.execute(
                "INSERT OR REPLACE INTO product_reviews (product_id, review_count, avg_rating) "
                "VALUES (?, ?, ?)",
                (pid, review_counts[pid], avg_ratings.get(pid, 0.0)),
            )
        self.conn.commit()

    def get_all_product_ids(self, domain: str) -> list[str]:
        """Return all product IDs for the given domain."""
        if self.conn is None:
            return []
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM products WHERE domain = ?", (domain,))
        return [row[0] for row in cur.fetchall()]

    def get_product_name(self, product_id: str) -> str:
        """Return product name for the given ID."""
        if self.conn is None:
            return ""
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM products WHERE id = ?", (product_id,))
        row = cur.fetchone()
        return row[0] if row else ""

    def get_product_info(self, product_id: str) -> dict:
        """Return product metadata dict."""
        if self.conn is None:
            return {}
        cur = self.conn.cursor()
        cur.execute(
            "SELECT name, brand, category, metadata FROM products WHERE id = ?",
            (product_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        return {
            "name": row[0],
            "brand": row[1],
            "category": row[2],
            "metadata": json.loads(row[3]) if row[3] else {},
        }

    def get_attribute_scores(self, product_id: str) -> dict[str, dict]:
        """Return attribute scores for a product.

        Returns:
            {aspect: {avg_sentiment, mention_count, confidence}, ...}
        """
        if self.conn is None:
            return {}
        cur = self.conn.cursor()
        cur.execute(
            "SELECT aspect, avg_sentiment, mention_count, confidence "
            "FROM product_attribute_scores WHERE product_id = ?",
            (product_id,),
        )
        result = {}
        for row in cur.fetchall():
            result[row[0]] = {
                "avg_sentiment": row[1],
                "mention_count": row[2],
                "confidence": row[3],
            }
        return result

    def get_review_stats(self, product_id: str) -> dict:
        """Return review count and average rating."""
        if self.conn is None:
            return {"review_count": 0, "avg_rating": 0.0}
        cur = self.conn.cursor()
        cur.execute(
            "SELECT review_count, avg_rating FROM product_reviews WHERE product_id = ?",
            (product_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"review_count": 0, "avg_rating": 0.0}
        return {"review_count": row[0], "avg_rating": row[1]}

    def get_total_aspects(self, domain: str) -> int:
        """Return total number of distinct aspects across all products in domain."""
        if self.conn is None:
            return 0
        cur = self.conn.cursor()
        cur.execute(
            "SELECT COUNT(DISTINCT aspect) FROM product_attribute_scores pas "
            "JOIN products p ON pas.product_id = p.id WHERE p.domain = ?",
            (domain,),
        )
        row = cur.fetchone()
        return row[0] if row else 0
