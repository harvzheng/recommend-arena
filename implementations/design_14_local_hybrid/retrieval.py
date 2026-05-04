"""Retrieval orchestration for design 14: FTS5 (Rust) + vector (Python).

Ingestion: builds an on-disk SQLite database with an FTS5 index over
`product_text || review_text` per product, and computes a vector for
each product. Query: runs both retrievers in parallel, returns ranked
ID lists for the RRF stage.

Vector encoder: lazy-imported sentence-transformers. The off-the-shelf
model is `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, MTEB top-tier at this
size). Override via `RECOMMEND_EMBED_MODEL_ST` env var. If
sentence-transformers isn't installed, the vector track is skipped and
the pipeline falls back to lexical-only — which is still a respectable
baseline (#5 / NDCG@5 0.518) and lets the rest of the pipeline work.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded; populated on first ingest.
_st_model_cache: dict[str, Any] = {}

DEFAULT_EMBED_MODEL = os.environ.get(
    "RECOMMEND_EMBED_MODEL_ST", "Qwen/Qwen3-Embedding-0.6B"
)


def _load_st_model(model_name: str):
    """Load (and cache) a SentenceTransformer model. Returns None if the
    library isn't available — caller should degrade to lexical-only."""
    if model_name in _st_model_cache:
        return _st_model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        logger.warning(
            "design_14: sentence-transformers not installed; "
            "vector track disabled. Pipeline will run lexical-only."
        )
        _st_model_cache[model_name] = None
        return None
    logger.info("design_14: loading embedding model %s …", model_name)
    model = SentenceTransformer(model_name)
    _st_model_cache[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# We deliberately do NOT reuse design_05's schema directly — design 14
# uses its own SQLite database file per domain so the prefilter and
# retrieval don't fight over the design 05 instance. The shape is
# compatible: same column names where they overlap, so the Rust
# prefilter SQL fragments work unchanged.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    brand TEXT,
    category TEXT
);

CREATE TABLE IF NOT EXISTS attribute_defs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    data_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS product_attributes (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    attribute_def_id INTEGER NOT NULL REFERENCES attribute_defs(id),
    value_numeric REAL,
    value_text TEXT
);

CREATE INDEX IF NOT EXISTS ix_pa_product ON product_attributes(product_id);
CREATE INDEX IF NOT EXISTS ix_pa_def ON product_attributes(attribute_def_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS reviews_fts USING fts5(
    content,
    product_id UNINDEXED,
    tokenize='porter unicode61'
);
"""


def open_db(path: str) -> sqlite3.Connection:
    """Open (or create) the per-domain SQLite database for design 14."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    try:
        conn.executescript(_FTS_SCHEMA)
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_products(
    conn: sqlite3.Connection,
    products: list[dict],
    reviews: list[dict],
) -> dict[str, int]:
    """Populate products / attributes / FTS5 from one domain's data.

    Returns a map external_id → internal_id for downstream callers.

    This is idempotent across re-runs: the existing rows for products
    that show up again are deleted first so an in-memory test that
    re-ingests the same dataset doesn't accumulate duplicates.
    """
    # Wipe existing rows. With a small catalog this is cheaper than
    # diff-and-update logic and keeps the code easy to reason about.
    conn.execute("DELETE FROM product_attributes")
    conn.execute("DELETE FROM products")
    conn.execute("DELETE FROM reviews_fts")

    # Group reviews by product
    reviews_by: dict[str, list[str]] = {}
    for r in reviews:
        pid = r.get("product_id") or r.get("id") or ""
        text = r.get("text") or r.get("review_text") or ""
        if pid and text:
            reviews_by.setdefault(pid, []).append(text)

    # Materialize attribute definitions on demand
    attr_def_ids: dict[tuple[str, str], int] = {}

    def def_id(name: str, data_type: str) -> int:
        key = (name, data_type)
        if key in attr_def_ids:
            return attr_def_ids[key]
        cur = conn.execute(
            "SELECT id FROM attribute_defs WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO attribute_defs (name, data_type) VALUES (?, ?)",
                (name, data_type),
            )
            attr_def_ids[key] = cur.lastrowid  # type: ignore
        else:
            attr_def_ids[key] = row[0]
        return attr_def_ids[key]

    id_map: dict[str, int] = {}
    for p in products:
        ext_id = p.get("id") or p.get("product_id") or ""
        if not ext_id:
            continue
        cur = conn.execute(
            "INSERT INTO products (external_id, name, brand, category) VALUES (?, ?, ?, ?)",
            (
                ext_id,
                p.get("name") or p.get("product_name") or ext_id,
                p.get("brand"),
                p.get("category"),
            ),
        )
        internal_id = cur.lastrowid
        id_map[ext_id] = internal_id  # type: ignore

        # Attributes
        attrs = p.get("attributes") or {}
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, (int, float)):
                conn.execute(
                    "INSERT INTO product_attributes "
                    "(product_id, attribute_def_id, value_numeric) VALUES (?, ?, ?)",
                    (internal_id, def_id(attr_name, "numeric"), float(attr_value)),
                )
            elif isinstance(attr_value, str):
                conn.execute(
                    "INSERT INTO product_attributes "
                    "(product_id, attribute_def_id, value_text) VALUES (?, ?, ?)",
                    (internal_id, def_id(attr_name, "text"), attr_value),
                )
            elif isinstance(attr_value, (list, tuple)):
                # Categorical multi-value: store joined for LIKE matching.
                conn.execute(
                    "INSERT INTO product_attributes "
                    "(product_id, attribute_def_id, value_text) VALUES (?, ?, ?)",
                    (
                        internal_id,
                        def_id(attr_name, "categorical"),
                        " ".join(str(v) for v in attr_value),
                    ),
                )

        # Specs as numeric attributes (where possible).
        specs = p.get("specs") or {}
        for spec_name, spec_value in specs.items():
            if isinstance(spec_value, (int, float)):
                conn.execute(
                    "INSERT INTO product_attributes "
                    "(product_id, attribute_def_id, value_numeric) VALUES (?, ?, ?)",
                    (internal_id, def_id(spec_name, "numeric"), float(spec_value)),
                )
            elif isinstance(spec_value, str):
                conn.execute(
                    "INSERT INTO product_attributes "
                    "(product_id, attribute_def_id, value_text) VALUES (?, ?, ?)",
                    (internal_id, def_id(spec_name, "text"), spec_value),
                )

        # FTS5 — one row per product, content = name + brand + category +
        # joined reviews. BM25 weighting: we index it all into a single
        # column for simplicity; the per-field weight tuning called out in
        # the spec (title=10, attributes=5, reviews=1) is a future lift.
        text_parts: list[str] = [p.get("name") or "", p.get("brand") or "", p.get("category") or ""]
        text_parts.extend(reviews_by.get(ext_id, []))
        text_parts.append(" ".join(str(v) for v in (p.get("attributes") or {}).values()
                                    if isinstance(v, (str, list, tuple))))
        content = " ".join(t for t in text_parts if t).strip()
        conn.execute(
            "INSERT INTO reviews_fts (content, product_id) VALUES (?, ?)",
            (content, ext_id),
        )

    conn.commit()
    return id_map


def encode_products(
    products: list[dict],
    reviews: list[dict],
    model_name: str = DEFAULT_EMBED_MODEL,
) -> dict[str, list[float]]:
    """Encode each product into a vector using sentence-transformers.

    Returns {external_id: vector}. Empty dict if the encoder is
    unavailable; caller should degrade to lexical-only retrieval.
    """
    model = _load_st_model(model_name)
    if model is None:
        return {}

    reviews_by: dict[str, list[str]] = {}
    for r in reviews:
        pid = r.get("product_id") or r.get("id") or ""
        text = r.get("text") or r.get("review_text") or ""
        if pid and text:
            reviews_by.setdefault(pid, []).append(text)

    texts: list[str] = []
    ids: list[str] = []
    for p in products:
        ext_id = p.get("id") or p.get("product_id") or ""
        if not ext_id:
            continue
        parts = [
            p.get("name") or "",
            p.get("brand") or "",
            p.get("category") or "",
        ]
        attrs = p.get("attributes") or {}
        attr_text = " ".join(
            f"{k}: {v}" for k, v in sorted(attrs.items()) if isinstance(v, (int, float, str))
        )
        if attr_text:
            parts.append(attr_text)
        # Use up to 3 reviews per product to keep the encoded text short
        # enough that the encoder doesn't waste effort on the tail.
        for rev in reviews_by.get(ext_id, [])[:3]:
            parts.append(rev[:300])
        texts.append(". ".join(p for p in parts if p))
        ids.append(ext_id)

    if not texts:
        return {}
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return {ids[i]: list(map(float, vectors[i])) for i in range(len(ids))}


def encode_query(query_text: str, model_name: str = DEFAULT_EMBED_MODEL) -> list[float] | None:
    """Encode a single query. Returns None when the encoder isn't loaded."""
    model = _load_st_model(model_name)
    if model is None:
        return None
    vec = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
    return list(map(float, vec[0]))


def cosine_top(
    query_vec: list[float],
    product_vecs: dict[str, list[float]],
    top_k: int = 100,
    candidate_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Score query against product vectors, return top_k. Caller pre-normalizes."""
    if not query_vec or not product_vecs:
        return []
    candidates = candidate_ids if candidate_ids is not None else list(product_vecs.keys())
    qv = query_vec
    qv_norm = math.sqrt(sum(x * x for x in qv)) or 1.0
    scored: list[tuple[str, float]] = []
    for pid in candidates:
        pv = product_vecs.get(pid)
        if pv is None:
            continue
        # Vectors are pre-normalized, but the query came through the
        # SentenceTransformer normalize_embeddings=True path too. Compute
        # dot product (== cosine on unit vectors).
        s = sum(a * b for a, b in zip(qv, pv)) / qv_norm
        scored.append((pid, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
