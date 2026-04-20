"""SQLite + ChromaDB storage layer for the multi-agent recommender."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from typing import Any

import chromadb
from chromadb.config import Settings

from .state import ProductAttributes

logger = logging.getLogger(__name__)


class Store:
    """Manages SQLite (structured data) and ChromaDB (vector search) storage."""

    def __init__(self, storage_dir: str | None = None, llm_provider=None):
        if storage_dir is None:
            storage_dir = tempfile.mkdtemp(prefix="design06_")
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self._llm = llm_provider

        # SQLite
        self._db_path = os.path.join(storage_dir, "products.db")
        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_schema()

        # ChromaDB
        self._chroma = chromadb.EphemeralClient()
        self._collections: dict[str, Any] = {}

    def _init_schema(self) -> None:
        """Create SQLite tables if they don't exist."""
        cur = self._db.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                domain TEXT NOT NULL,
                brand TEXT DEFAULT '',
                category TEXT DEFAULT '',
                specs_json TEXT DEFAULT '{}',
                attributes_json TEXT DEFAULT '{}',
                sentiment_json TEXT DEFAULT '{}',
                review_count INTEGER DEFAULT 0,
                avg_rating REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS reviews_raw (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                reviewer TEXT DEFAULT '',
                rating REAL DEFAULT 0,
                review_text TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            );

            CREATE INDEX IF NOT EXISTS idx_products_domain ON products(domain);
            CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews_raw(product_id);
        """)
        self._db.commit()

    def _get_collection(self, domain: str):
        """Get or create a ChromaDB collection for a domain."""
        if domain not in self._collections:
            self._collections[domain] = self._chroma.get_or_create_collection(
                name=f"{domain}_products",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[domain]

    def clear_domain(self, domain: str) -> None:
        """Remove all data for a domain (for re-ingestion)."""
        cur = self._db.cursor()
        cur.execute(
            "DELETE FROM reviews_raw WHERE product_id IN "
            "(SELECT product_id FROM products WHERE domain = ?)",
            (domain,),
        )
        cur.execute("DELETE FROM products WHERE domain = ?", (domain,))
        self._db.commit()

        if domain in self._collections:
            self._chroma.delete_collection(f"{domain}_products")
            del self._collections[domain]

    def store_product(self, product: ProductAttributes, reviews: list[dict]) -> None:
        """Persist a product and its reviews to SQLite and ChromaDB."""
        cur = self._db.cursor()

        # Upsert product
        cur.execute(
            """INSERT OR REPLACE INTO products
               (product_id, product_name, domain, brand, category,
                specs_json, attributes_json, sentiment_json,
                review_count, avg_rating)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                product.product_id,
                product.product_name,
                product.domain,
                product.brand,
                product.category,
                json.dumps(product.specs),
                json.dumps(product.attributes, default=_json_default),
                json.dumps(product.sentiment_scores),
                product.review_count,
                product.avg_rating,
            ),
        )

        # Store reviews
        for review in reviews:
            cur.execute(
                "INSERT INTO reviews_raw (product_id, reviewer, rating, review_text) "
                "VALUES (?, ?, ?, ?)",
                (
                    product.product_id,
                    review.get("reviewer", ""),
                    review.get("rating", 0),
                    review.get("text", ""),
                ),
            )

        self._db.commit()

        # Embed into ChromaDB
        summary = self._build_summary(product)
        collection = self._get_collection(product.domain)

        # Generate embedding
        meta = {
            "product_name": product.product_name,
            "domain": product.domain,
            "category": product.category,
            "brand": product.brand,
        }
        if self._llm is not None:
            try:
                embedding = self._llm.embed(summary)
                collection.upsert(
                    ids=[product.product_id],
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[meta],
                )
                return
            except RuntimeError:
                logger.warning(
                    "Embedding failed for %s, falling back to document-only",
                    product.product_id,
                )

        # Fallback: let ChromaDB use its default embedding function
        collection.upsert(
            ids=[product.product_id],
            documents=[summary],
            metadatas=[meta],
        )

    def _build_summary(self, product: ProductAttributes) -> str:
        """Build a text summary of a product for embedding."""
        parts = [f"{product.product_name} by {product.brand}"]
        parts.append(f"Category: {product.category}")

        for attr, val in product.attributes.items():
            if isinstance(val, list):
                parts.append(f"{attr}: {', '.join(str(v) for v in val)}")
            else:
                parts.append(f"{attr}: {val}")

        for spec, val in product.specs.items():
            if isinstance(val, list):
                parts.append(f"{spec}: {', '.join(str(v) for v in val)}")
            else:
                parts.append(f"{spec}: {val}")

        if product.sentiment_scores:
            top_pos = sorted(
                product.sentiment_scores.items(), key=lambda x: -x[1]
            )[:3]
            if top_pos:
                parts.append(
                    "Positive sentiments: "
                    + ", ".join(f"{k} ({v:.1f})" for k, v in top_pos)
                )

        return ". ".join(parts)

    def query_by_domain(self, domain: str) -> list[ProductAttributes]:
        """Load all products for a domain from SQLite."""
        cur = self._db.cursor()
        cur.execute("SELECT * FROM products WHERE domain = ?", (domain,))
        rows = cur.fetchall()
        return [self._row_to_product(row) for row in rows]

    def query_by_filters(
        self, domain: str, filters: dict[str, Any]
    ) -> list[ProductAttributes]:
        """Query products with hard filters against specs and attributes."""
        all_products = self.query_by_domain(domain)
        if not filters:
            return all_products

        results = []
        for p in all_products:
            if self._matches_filters(p, filters):
                results.append(p)
        return results

    def _matches_filters(
        self, product: ProductAttributes, filters: dict[str, Any]
    ) -> bool:
        """Check if a product matches all hard filters."""
        for key, value in filters.items():
            # Check specs
            if key in product.specs:
                spec_val = product.specs[key]
                if isinstance(spec_val, list):
                    # For list specs (e.g. lengths_cm), check if any match
                    if isinstance(value, dict):
                        # Range filter: {"min": X, "max": Y}
                        if "min" in value and not any(
                            v >= value["min"] for v in spec_val
                        ):
                            return False
                        if "max" in value and not any(
                            v <= value["max"] for v in spec_val
                        ):
                            return False
                    elif value not in spec_val:
                        return False
                else:
                    if isinstance(value, dict):
                        if "min" in value and spec_val < value["min"]:
                            return False
                        if "max" in value and spec_val > value["max"]:
                            return False
                    elif spec_val != value:
                        return False
            # Check attributes
            elif key in product.attributes:
                attr_val = product.attributes[key]
                if isinstance(attr_val, list):
                    if isinstance(value, str) and value not in attr_val:
                        return False
                    elif isinstance(value, list) and not set(value) & set(attr_val):
                        return False
                else:
                    if isinstance(value, dict):
                        if "min" in value and attr_val < value["min"]:
                            return False
                        if "max" in value and attr_val > value["max"]:
                            return False
                    elif attr_val != value:
                        return False
            # Check category
            elif key == "category" and product.category != value:
                return False
            # Unknown filter key -- skip (don't reject)
        return True

    def vector_search(
        self, domain: str, query_text: str, n_results: int = 20
    ) -> list[str]:
        """Search ChromaDB for similar products by text, return product IDs."""
        collection = self._get_collection(domain)
        if collection.count() == 0:
            return []

        n = min(n_results, collection.count())
        if self._llm is not None:
            try:
                query_embedding = self._llm.embed(query_text)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n,
                )
            except RuntimeError:
                logger.warning("Embed failed for vector search, using text query")
                results = collection.query(
                    query_texts=[query_text],
                    n_results=n,
                )
        else:
            results = collection.query(
                query_texts=[query_text],
                n_results=n,
            )

        return results["ids"][0] if results["ids"] else []

    def get_products_by_ids(
        self, product_ids: list[str], domain: str
    ) -> list[ProductAttributes]:
        """Fetch specific products by ID."""
        if not product_ids:
            return []
        placeholders = ",".join("?" for _ in product_ids)
        cur = self._db.cursor()
        cur.execute(
            f"SELECT * FROM products WHERE product_id IN ({placeholders}) AND domain = ?",
            (*product_ids, domain),
        )
        rows = cur.fetchall()
        return [self._row_to_product(row) for row in rows]

    def _row_to_product(self, row: sqlite3.Row) -> ProductAttributes:
        """Convert a SQLite row to a ProductAttributes."""
        return ProductAttributes(
            product_id=row["product_id"],
            product_name=row["product_name"],
            domain=row["domain"],
            brand=row["brand"],
            category=row["category"],
            specs=json.loads(row["specs_json"]),
            attributes=json.loads(row["attributes_json"]),
            sentiment_scores=json.loads(row["sentiment_json"]),
            review_count=row["review_count"],
            avg_rating=row["avg_rating"],
        )


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
