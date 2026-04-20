"""SQLite FTS5-based faceted index.

This module implements a search index using SQLite FTS5 for text search
combined with structured tables for faceted filtering. This is the
zero-dependency fallback that replaces the Typesense backend from the
design document.
"""

from __future__ import annotations

import json
import logging
import sqlite3

logger = logging.getLogger(__name__)


class FacetedIndex:
    """SQLite-based faceted search index.

    Stores product documents in a structured table with an FTS5 virtual
    table for full-text search on text fields.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._collections: dict[str, dict] = {}

    def create_collection(self, collection_name: str, domain_config: dict) -> None:
        """Create tables for a domain collection.

        Creates:
            1. A main document table with typed columns for all facets
            2. An FTS5 virtual table for text search on name + review_summary
            3. A categorical facets table for multi-valued category fields
        """
        facets = domain_config["facets"]
        self._collections[collection_name] = domain_config

        # Build column definitions for the main table
        columns = [
            "id TEXT PRIMARY KEY",
            "name TEXT",
            "brand TEXT",
            "domain TEXT",
            "category TEXT",
            "review_count INTEGER DEFAULT 0",
            "review_summary TEXT DEFAULT ''",
            "popularity INTEGER DEFAULT 0",
        ]

        for fname, fdef in facets.items():
            ftype = fdef["type"]
            if ftype == "numeric":
                columns.append(f"{fname} REAL DEFAULT 0.5")
            elif ftype == "spec_numeric":
                columns.append(f"{fname} REAL")
            elif ftype == "boolean":
                columns.append(f"{fname} INTEGER DEFAULT 0")
            # categorical fields go in a separate table

        col_sql = ", ".join(columns)
        self.conn.execute(f"DROP TABLE IF EXISTS {collection_name}")
        self.conn.execute(f"CREATE TABLE {collection_name} ({col_sql})")

        # FTS5 for text search on name and review_summary
        self.conn.execute(f"DROP TABLE IF EXISTS {collection_name}_fts")
        self.conn.execute(
            f"CREATE VIRTUAL TABLE {collection_name}_fts "
            f"USING fts5(id UNINDEXED, name, review_summary, brand, "
            f"content={collection_name}, content_rowid=rowid)"
        )

        # Categorical facets table (many-to-many)
        self.conn.execute(f"DROP TABLE IF EXISTS {collection_name}_cats")
        self.conn.execute(
            f"CREATE TABLE {collection_name}_cats ("
            f"  product_id TEXT, facet_name TEXT, facet_value TEXT, "
            f"  PRIMARY KEY (product_id, facet_name, facet_value))"
        )

        # Lengths table for spec arrays (e.g. available ski lengths)
        self.conn.execute(f"DROP TABLE IF EXISTS {collection_name}_lengths")
        self.conn.execute(
            f"CREATE TABLE {collection_name}_lengths ("
            f"  product_id TEXT, length_cm REAL, "
            f"  PRIMARY KEY (product_id, length_cm))"
        )

        self.conn.commit()

    def upsert_document(self, collection_name: str, doc: dict) -> None:
        """Insert or replace a document in the collection."""
        domain_config = self._collections[collection_name]
        facets = domain_config["facets"]

        # Build the main table insert
        main_cols = ["id", "name", "brand", "domain", "category",
                     "review_count", "review_summary", "popularity"]
        main_vals = [
            doc.get("id", ""),
            doc.get("name", ""),
            doc.get("brand", ""),
            doc.get("domain", ""),
            doc.get("category", ""),
            doc.get("review_count", 0),
            doc.get("review_summary", ""),
            doc.get("popularity", 0),
        ]

        for fname, fdef in facets.items():
            ftype = fdef["type"]
            if ftype in ("numeric", "spec_numeric"):
                main_cols.append(fname)
                main_vals.append(doc.get(fname))
            elif ftype == "boolean":
                main_cols.append(fname)
                main_vals.append(1 if doc.get(fname) else 0)

        placeholders = ", ".join(["?"] * len(main_cols))
        col_names = ", ".join(main_cols)
        self.conn.execute(
            f"INSERT OR REPLACE INTO {collection_name} ({col_names}) "
            f"VALUES ({placeholders})",
            main_vals,
        )

        # Insert categorical facets
        product_id = doc.get("id", "")
        self.conn.execute(
            f"DELETE FROM {collection_name}_cats WHERE product_id = ?",
            (product_id,),
        )
        for fname, fdef in facets.items():
            if fdef["type"] == "categorical":
                values = doc.get(fname, [])
                if isinstance(values, str):
                    values = [values]
                for val in values:
                    self.conn.execute(
                        f"INSERT OR IGNORE INTO {collection_name}_cats "
                        f"(product_id, facet_name, facet_value) VALUES (?, ?, ?)",
                        (product_id, fname, val),
                    )

        # Insert available lengths
        self.conn.execute(
            f"DELETE FROM {collection_name}_lengths WHERE product_id = ?",
            (product_id,),
        )
        lengths = doc.get("lengths_cm", [])
        for length in lengths:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {collection_name}_lengths "
                f"(product_id, length_cm) VALUES (?, ?)",
                (product_id, float(length)),
            )

        # Update FTS index
        # First delete old entry, then insert new
        rowid = self.conn.execute(
            f"SELECT rowid FROM {collection_name} WHERE id = ?",
            (product_id,),
        ).fetchone()
        if rowid:
            self.conn.execute(
                f"INSERT INTO {collection_name}_fts(rowid, id, name, review_summary, brand) "
                f"VALUES (?, ?, ?, ?, ?)",
                (rowid[0], product_id, doc.get("name", ""),
                 doc.get("review_summary", ""), doc.get("brand", "")),
            )

        self.conn.commit()

    def search(
        self,
        collection_name: str,
        text_query: str | None = None,
        filters: dict | None = None,
        sort_fields: list[tuple[str, str]] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Execute a faceted search.

        Args:
            collection_name: Collection to search.
            text_query: FTS5 query string, or None/"*" for no text filter.
            filters: Dict of {field: (operator, value)} for facet filtering.
            sort_fields: List of (field, "asc"|"desc") for sorting.
            limit: Maximum results.

        Returns:
            List of matching document dicts with _text_score added.
        """
        domain_config = self._collections.get(collection_name)
        if not domain_config:
            return []

        has_text = text_query and text_query.strip() != "*"

        # Build the query
        if has_text:
            # Use FTS5 for text search, join with main table
            fts_query = self._sanitize_fts_query(text_query)
            sql = (
                f"SELECT m.*, fts.rank AS _text_rank "
                f"FROM {collection_name} m "
                f"JOIN {collection_name}_fts fts ON m.rowid = fts.rowid "
                f"WHERE {collection_name}_fts MATCH ?"
            )
            params: list = [fts_query]
        else:
            sql = f"SELECT m.*, 0 AS _text_rank FROM {collection_name} m WHERE 1=1"
            params = []

        # Apply facet filters
        if filters:
            facets = domain_config["facets"]
            for field, (op, value) in filters.items():
                if field in facets and facets[field]["type"] == "categorical":
                    # Filter via the categorical table
                    sql += (
                        f" AND m.id IN (SELECT product_id FROM {collection_name}_cats "
                        f"WHERE facet_name = ? AND facet_value = ?)"
                    )
                    params.extend([field, value])
                elif field == "length_cm":
                    # Filter via lengths table
                    sql_op = {">=": ">=", "<=": "<=", ">": ">", "<": "<", "=": "="}.get(op, "=")
                    sql += (
                        f" AND m.id IN (SELECT product_id FROM {collection_name}_lengths "
                        f"WHERE length_cm {sql_op} ?)"
                    )
                    params.append(float(value))
                elif field in facets and facets[field]["type"] == "boolean":
                    bool_val = 1 if value.lower() in ("true", "1", "yes") else 0
                    sql += f" AND m.{field} = ?"
                    params.append(bool_val)
                else:
                    # Numeric / spec_numeric fields on the main table
                    sql_op = {">=": ">=", "<=": "<=", ">": ">", "<": "<", "=": "="}.get(op, "=")
                    try:
                        sql += f" AND m.{field} {sql_op} ?"
                        params.append(float(value))
                    except ValueError:
                        pass

        # Sorting
        if sort_fields:
            order_parts = []
            if has_text:
                order_parts.append("fts.rank")  # FTS5 rank (lower = better match)
            for field, direction in sort_fields:
                order_parts.append(f"m.{field} {direction.upper()}")
            sql += " ORDER BY " + ", ".join(order_parts)
        elif has_text:
            sql += " ORDER BY fts.rank"
        else:
            sql += " ORDER BY m.popularity DESC"

        sql += f" LIMIT {limit}"

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("Search query failed: %s — falling back to full scan", e)
            # Fallback: return all products from the main table
            rows = self.conn.execute(
                f"SELECT *, 0 AS _text_rank FROM {collection_name} "
                f"ORDER BY popularity DESC LIMIT {limit}"
            ).fetchall()

        # Convert to dicts and attach categorical facets
        results = []
        for row in rows:
            doc = dict(row)
            # Fetch categorical facets for this product
            cat_rows = self.conn.execute(
                f"SELECT facet_name, facet_value FROM {collection_name}_cats "
                f"WHERE product_id = ?",
                (doc["id"],),
            ).fetchall()
            for cr in cat_rows:
                fn = cr["facet_name"]
                if fn not in doc or not isinstance(doc.get(fn), list):
                    doc[fn] = []
                doc[fn].append(cr["facet_value"])

            # Fetch available lengths
            len_rows = self.conn.execute(
                f"SELECT length_cm FROM {collection_name}_lengths "
                f"WHERE product_id = ?",
                (doc["id"],),
            ).fetchall()
            doc["lengths_cm"] = [lr["length_cm"] for lr in len_rows]

            results.append(doc)

        return results

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a query string for FTS5.

        Removes special FTS5 operators and wraps terms so they
        work as a simple OR search.
        """
        # Remove FTS5 special characters
        import re
        cleaned = re.sub(r'[^\w\s\-]', ' ', query)
        terms = cleaned.split()
        if not terms:
            return '""'
        # Join with OR for broader matching
        return " OR ".join(f'"{t}"' for t in terms if t)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
