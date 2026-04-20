"""Posterior distribution management for the Bayesian recommender.

Each product maintains a ProductBelief containing posterior hyperparameters
for every attribute in the domain schema.  Posteriors are stored as plain
dicts so they serialize directly to JSON for SQLite persistence.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .schema import AttributeSpec, get_schema


# ---------------------------------------------------------------------------
# ProductBelief data class
# ---------------------------------------------------------------------------

@dataclass
class ProductBelief:
    """Posterior belief state for a single product."""

    product_id: str
    domain: str
    name: str
    posteriors: dict[str, dict]   # attr_name -> posterior hyperparams
    evidence_count: int = 0
    embedding: list[float] | None = None  # Dense embedding vector

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def posteriors_json(self) -> str:
        return json.dumps(self.posteriors)

    @staticmethod
    def posteriors_from_json(blob: str) -> dict[str, dict]:
        return json.loads(blob)


# ---------------------------------------------------------------------------
# Initialize posteriors from a schema (copy priors)
# ---------------------------------------------------------------------------

def init_posteriors(schema: list[AttributeSpec]) -> dict[str, dict]:
    """Create an initial posterior dict by copying each attribute's prior."""
    posteriors: dict[str, dict] = {}
    for spec in schema:
        # Deep-copy alpha list so mutations don't affect the schema
        prior_copy = {}
        for k, v in spec.prior.items():
            prior_copy[k] = list(v) if isinstance(v, list) else v
        posteriors[spec.name] = prior_copy
    return posteriors


# ---------------------------------------------------------------------------
# SQLite-backed belief store
# ---------------------------------------------------------------------------

class BeliefStore:
    """Manages product beliefs in an in-memory SQLite database."""

    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                name TEXT NOT NULL,
                posteriors JSON NOT NULL,
                evidence_count INTEGER DEFAULT 0,
                embedding JSON,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS reviews (
                review_id TEXT PRIMARY KEY,
                product_id TEXT,
                raw_text TEXT,
                extracted JSON,
                ingested_at TEXT
            );
        """)
        self.db.commit()

    def clear_domain(self, domain: str) -> None:
        """Remove all products and reviews for a domain (for re-ingestion)."""
        pids = [r[0] for r in self.db.execute(
            "SELECT product_id FROM products WHERE domain = ?", (domain,)
        ).fetchall()]
        if pids:
            placeholders = ",".join("?" * len(pids))
            self.db.execute(
                f"DELETE FROM reviews WHERE product_id IN ({placeholders})", pids
            )
            self.db.execute(
                "DELETE FROM products WHERE domain = ?", (domain,)
            )
            self.db.commit()

    def upsert_belief(self, belief: ProductBelief) -> None:
        now = datetime.now(timezone.utc).isoformat()
        embedding_json = json.dumps(belief.embedding) if belief.embedding else None
        self.db.execute("""
            INSERT INTO products (product_id, domain, name, posteriors, evidence_count, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                posteriors = excluded.posteriors,
                evidence_count = excluded.evidence_count,
                embedding = excluded.embedding,
                updated_at = excluded.updated_at
        """, (
            belief.product_id,
            belief.domain,
            belief.name,
            belief.posteriors_json(),
            belief.evidence_count,
            embedding_json,
            now,
        ))
        self.db.commit()

    def store_review(self, review_id: str, product_id: str,
                     raw_text: str, extracted: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute("""
            INSERT OR REPLACE INTO reviews (review_id, product_id, raw_text, extracted, ingested_at)
            VALUES (?, ?, ?, ?, ?)
        """, (review_id, product_id, raw_text, json.dumps(extracted), now))
        self.db.commit()

    def get_belief(self, product_id: str) -> ProductBelief | None:
        row = self.db.execute(
            "SELECT product_id, domain, name, posteriors, evidence_count, embedding "
            "FROM products WHERE product_id = ?",
            (product_id,),
        ).fetchone()
        if row is None:
            return None
        return ProductBelief(
            product_id=row[0],
            domain=row[1],
            name=row[2],
            posteriors=ProductBelief.posteriors_from_json(row[3]),
            evidence_count=row[4],
            embedding=json.loads(row[5]) if row[5] else None,
        )

    def get_all_beliefs(self, domain: str) -> list[ProductBelief]:
        rows = self.db.execute(
            "SELECT product_id, domain, name, posteriors, evidence_count, embedding "
            "FROM products WHERE domain = ?",
            (domain,),
        ).fetchall()
        return [
            ProductBelief(
                product_id=r[0],
                domain=r[1],
                name=r[2],
                posteriors=ProductBelief.posteriors_from_json(r[3]),
                evidence_count=r[4],
                embedding=json.loads(r[5]) if r[5] else None,
            )
            for r in rows
        ]

    def get_review_texts(self, product_id: str) -> list[str]:
        """Return all raw review texts for a product."""
        rows = self.db.execute(
            "SELECT raw_text FROM reviews WHERE product_id = ?",
            (product_id,),
        ).fetchall()
        return [r[0] for r in rows]
