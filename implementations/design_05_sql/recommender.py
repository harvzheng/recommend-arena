"""Main Recommender implementation for Design 05: SQL-First / SQLite + FTS5.

Implements the shared Recommender protocol using:
- SQLite for all data storage
- FTS5 for full-text search with BM25
- LLM for attribute extraction (ingestion) and query parsing (query time)
- Three-signal ranking: attribute match + BM25 + sentiment
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path

# Add shared module to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .extraction import ingest_product_with_specs, ingest_review
from .query_parser import parse_query
from .ranking import (
    W_ATTRIBUTE,
    W_EMBEDDING,
    W_FTS,
    W_SENTIMENT,
    build_explanation,
    compute_attribute_scores,
    compute_embedding_scores,
    compute_fts_scores,
    compute_sentiment_scores,
)
from .schema import ensure_domain, init_db
from .synonyms import expand_synonyms, pre_expand_query, seed_synonyms

logger = logging.getLogger(__name__)


class SqlRecommender:
    """Recommender backed by SQLite + FTS5.

    Uses LLM only for:
    1. Attribute extraction during ingestion (optional, since benchmark data
       already has ground-truth attributes)
    2. Query parsing at query time (single LLM call per query)

    All ranking is deterministic: attribute match + BM25 + sentiment.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        llm_provider: LLMProvider | None = None,
        use_llm_extraction: bool = False,
    ):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        init_db(self.db)

        self._llm_provider = llm_provider
        self._use_llm_extraction = use_llm_extraction

        # Cache for domain_id lookups
        self._domain_ids: dict[str, int] = {}
        # Cache for product_id mappings (external -> internal)
        self._product_id_maps: dict[str, dict[str, int]] = {}
        # Track if domain has been ingested
        self._ingested_domains: set[str] = set()
        # In-memory embedding store: internal_product_id -> embedding vector
        self._product_embeddings: dict[int, list[float]] = {}

    @property
    def llm(self) -> LLMProvider:
        if self._llm_provider is None:
            self._llm_provider = get_provider()
        return self._llm_provider

    def ingest(
        self,
        products: list[dict],
        reviews: list[dict],
        domain: str,
    ) -> None:
        """Ingest product and review data for a given domain.

        Products include ground-truth attributes from the benchmark data.
        Reviews are stored for FTS indexing and sentiment.
        """
        # If already ingested, clear and re-ingest
        if domain in self._ingested_domains:
            self._clear_domain(domain)

        domain_id = ensure_domain(self.db, domain)
        self._domain_ids[domain] = domain_id

        # Seed synonyms
        seed_synonyms(self.db, domain, domain_id)

        # Ingest products with their spec-derived attributes
        product_id_map: dict[str, int] = {}
        for product in products:
            ext_id = product.get("id", product.get("product_id", ""))
            internal_id = ingest_product_with_specs(self.db, product, domain_id)
            product_id_map[ext_id] = internal_id

        self._product_id_maps[domain] = product_id_map
        self.db.commit()

        # Ingest reviews
        for review in reviews:
            ingest_review(
                self.db,
                self._llm_provider if self._use_llm_extraction else None,
                review,
                domain,
                domain_id,
                product_id_map,
                use_llm=self._use_llm_extraction,
            )

        self.db.commit()

        # Generate embeddings for all products in this domain
        self._generate_product_embeddings(domain_id)

        self._ingested_domains.add(domain)

        logger.info(
            "Ingested %d products and %d reviews for domain '%s'",
            len(products), len(reviews), domain,
        )

    def query(
        self,
        query_text: str,
        domain: str,
        top_k: int = 10,
    ) -> list[RecommendationResult]:
        """Query the recommendation system with natural language.

        Pipeline:
        1. LLM parses query into structured filters + keywords
        2. SQL retrieves candidate products matching filters
        3. Python ranks candidates using three signals
        4. Returns top_k results with explanations
        """
        domain_id = self._domain_ids.get(domain)
        if domain_id is None:
            domain_id = self._get_domain_id(domain)
            if domain_id is None:
                logger.warning("Domain '%s' not found", domain)
                return []

        # Stage 1: LLM query parse (single LLM call)
        parsed = parse_query(self.llm, self.db, query_text, domain, domain_id)
        logger.debug("Parsed query: %s", json.dumps(parsed, indent=2))

        # Expand keywords with synonyms before FTS query
        if parsed.get("keywords"):
            parsed["keywords"] = expand_synonyms(
                self.db, parsed["keywords"], domain_id
            )
            logger.debug("Expanded keywords: %s", parsed["keywords"])

        # Stage 2: Get candidate products
        candidates = self._get_candidates(domain_id, parsed)

        if not candidates:
            # Fallback: return all products in domain, ranked by FTS + sentiment
            candidates = self._get_all_products(domain_id)

        if not candidates:
            return []

        product_ids = [c["id"] for c in candidates]

        # Stage 3: Rank candidates
        # Compute FTS scores for all candidates at once
        fts_scores = compute_fts_scores(self.db, product_ids, parsed.get("keywords", ""))

        # Compute embedding similarity scores
        query_embedding = None
        try:
            query_embedding = self.llm.embed(query_text)
        except Exception as e:
            logger.warning("Failed to embed query: %s", e)

        embedding_scores = compute_embedding_scores(
            query_embedding or [], self._product_embeddings, product_ids
        )

        # Compute sentiment scores for all candidates
        sentiment_scores = compute_sentiment_scores(self.db, product_ids)

        # Score each candidate
        scored_results = []
        for candidate in candidates:
            pid = candidate["id"]

            # Attribute match scores
            attr_scores = compute_attribute_scores(
                self.db, pid, domain_id, parsed["filters"]
            )

            fts_score = fts_scores.get(pid, 0.0)
            embed_score = embedding_scores.get(pid, 0.0)
            sent_score = sentiment_scores.get(pid, 0.5)

            # Compute weighted total
            if attr_scores:
                avg_attr = sum(attr_scores.values()) / len(attr_scores)
            else:
                avg_attr = 0.0

            total = (
                W_ATTRIBUTE * avg_attr
                + W_FTS * fts_score
                + W_EMBEDDING * embed_score
                + W_SENTIMENT * sent_score
            )

            # Clamp to [0, 1]
            total = max(0.0, min(1.0, total))

            explanation = build_explanation(
                attr_scores, fts_score, embed_score, sent_score, total, parsed["filters"]
            )

            scored_results.append(RecommendationResult(
                product_id=candidate["external_id"],
                product_name=candidate["name"],
                score=total,
                explanation=explanation,
                matched_attributes=attr_scores,
            ))

        # Sort by score descending
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def _generate_product_embeddings(self, domain_id: int) -> None:
        """Generate embedding vectors for all products in a domain.

        Creates a text representation combining product name, category, key
        attributes, and a sample of review content, then calls embed().
        """
        products = self.db.execute(
            "SELECT id, name, brand, category FROM products WHERE domain_id = ?",
            (domain_id,),
        ).fetchall()

        for product in products:
            pid = product["id"]
            name = product["name"] or ""
            brand = product["brand"] or ""
            category = product["category"] or ""

            # Gather attribute values for this product
            attrs = self.db.execute(
                "SELECT ad.name, pa.value_numeric, pa.value_text "
                "FROM product_attributes pa "
                "JOIN attribute_defs ad ON ad.id = pa.attribute_def_id "
                "WHERE pa.product_id = ?",
                (pid,),
            ).fetchall()

            attr_parts = []
            for a in attrs:
                aname = a["name"]
                if a["value_numeric"] is not None:
                    attr_parts.append(f"{aname}: {a['value_numeric']}")
                elif a["value_text"]:
                    attr_parts.append(f"{aname}: {a['value_text']}")

            # Gather review snippets (first 200 chars of up to 3 reviews)
            reviews = self.db.execute(
                "SELECT content FROM reviews WHERE product_id = ? LIMIT 3",
                (pid,),
            ).fetchall()
            review_snippets = " ".join(
                (r["content"] or "")[:200] for r in reviews
            )

            # Build embedding text
            embed_text = f"{name} {brand} {category}. {' '.join(attr_parts)}. {review_snippets}"
            # Truncate to reasonable length for embedding
            embed_text = embed_text[:1000]

            try:
                embedding = self.llm.embed(embed_text)
                self._product_embeddings[pid] = embedding
            except Exception as e:
                logger.warning("Failed to embed product %s: %s", name, e)

        logger.info(
            "Generated embeddings for %d/%d products",
            len([p for p in products if p["id"] in self._product_embeddings]),
            len(products),
        )

    def _get_candidates(
        self,
        domain_id: int,
        parsed: dict,
    ) -> list[dict]:
        """Get candidate products using SQL filters.

        Uses a relaxed approach: products must match at least some filters,
        and near-misses are included (scoring handles the ranking).
        """
        filters = parsed.get("filters", [])

        if not filters:
            # No structured filters: return all products, rely on FTS + sentiment
            return self._get_all_products(domain_id)

        # Build a query that finds products matching ANY filter
        # (inclusive retrieval, let ranking sort out quality)
        # We use a scoring approach: count how many filters each product matches
        conditions = []
        params: list = []

        for f in filters:
            attr_name = f["attribute"]
            op = f["op"]
            value = f["value"]

            # Get attribute_def_id
            attr_def = self.db.execute(
                "SELECT id, data_type FROM attribute_defs "
                "WHERE domain_id = ? AND name = ?",
                (domain_id, attr_name),
            ).fetchone()

            if attr_def is None:
                continue

            attr_def_id = attr_def["id"]
            data_type = attr_def["data_type"]

            if op in ("gte", "lte", "eq") and data_type in ("numeric", "scale"):
                # For near-miss tolerance, widen the range
                try:
                    target_val = float(value)
                except (TypeError, ValueError):
                    continue

                if op == "gte":
                    # Allow 20% below target for near-misses
                    if data_type == "scale":
                        tolerance = 2  # 2 points on a 1-10 scale
                    else:
                        tolerance = target_val * 0.15
                    conditions.append(
                        f"EXISTS (SELECT 1 FROM product_attributes pa "
                        f"WHERE pa.product_id = p.id AND pa.attribute_def_id = ? "
                        f"AND pa.value_numeric >= ?)"
                    )
                    params.extend([attr_def_id, target_val - tolerance])
                elif op == "lte":
                    if data_type == "scale":
                        tolerance = 2
                    else:
                        tolerance = abs(target_val * 0.15)
                    conditions.append(
                        f"EXISTS (SELECT 1 FROM product_attributes pa "
                        f"WHERE pa.product_id = p.id AND pa.attribute_def_id = ? "
                        f"AND pa.value_numeric <= ?)"
                    )
                    params.extend([attr_def_id, target_val + tolerance])
                elif op == "eq":
                    conditions.append(
                        f"EXISTS (SELECT 1 FROM product_attributes pa "
                        f"WHERE pa.product_id = p.id AND pa.attribute_def_id = ? "
                        f"AND ABS(pa.value_numeric - ?) < 0.5)"
                    )
                    params.extend([attr_def_id, target_val])

            elif op == "contains":
                conditions.append(
                    f"EXISTS (SELECT 1 FROM product_attributes pa "
                    f"WHERE pa.product_id = p.id AND pa.attribute_def_id = ? "
                    f"AND pa.value_text LIKE ?)"
                )
                params.extend([attr_def_id, f"%{value}%"])

            elif op == "not_contains":
                conditions.append(
                    f"NOT EXISTS (SELECT 1 FROM product_attributes pa "
                    f"WHERE pa.product_id = p.id AND pa.attribute_def_id = ? "
                    f"AND pa.value_text LIKE ?)"
                )
                params.extend([attr_def_id, f"%{value}%"])

        if not conditions:
            return self._get_all_products(domain_id)

        # Use OR to be inclusive (near-misses included)
        # Products matching more conditions will score higher in ranking
        where_clause = " OR ".join(conditions)

        sql = f"""
            SELECT p.id, p.external_id, p.name, p.brand, p.category
            FROM products p
            WHERE p.domain_id = ?
              AND ({where_clause})
            ORDER BY p.name
        """
        params_full = [domain_id] + params

        rows = self.db.execute(sql, params_full).fetchall()

        # If too few results from filtered query, fall back to all products
        if len(rows) < 3:
            return self._get_all_products(domain_id)

        return [dict(row) for row in rows]

    def _get_all_products(self, domain_id: int) -> list[dict]:
        """Get all products in a domain."""
        rows = self.db.execute(
            "SELECT id, external_id, name, brand, category "
            "FROM products WHERE domain_id = ?",
            (domain_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _get_domain_id(self, domain: str) -> int | None:
        """Look up domain_id from the database."""
        row = self.db.execute(
            "SELECT id FROM domains WHERE name = ?", (domain,)
        ).fetchone()
        if row:
            domain_id = row[0]
            self._domain_ids[domain] = domain_id
            return domain_id
        return None

    def _clear_domain(self, domain: str) -> None:
        """Clear all data for a domain to allow re-ingestion."""
        domain_id = self._domain_ids.get(domain)
        if domain_id is None:
            return

        # Delete FTS entries for reviews being removed
        self.db.execute(
            "DELETE FROM reviews_fts WHERE rowid IN "
            "(SELECT id FROM reviews WHERE product_id IN "
            "(SELECT id FROM products WHERE domain_id = ?))",
            (domain_id,),
        )

        # Delete in correct order for foreign keys
        self.db.execute(
            "DELETE FROM reviews WHERE product_id IN "
            "(SELECT id FROM products WHERE domain_id = ?)",
            (domain_id,),
        )
        self.db.execute(
            "DELETE FROM product_attributes WHERE product_id IN "
            "(SELECT id FROM products WHERE domain_id = ?)",
            (domain_id,),
        )
        self.db.execute("DELETE FROM products WHERE domain_id = ?", (domain_id,))
        self.db.execute("DELETE FROM synonyms WHERE domain_id = ?", (domain_id,))
        self.db.execute("DELETE FROM attribute_defs WHERE domain_id = ?", (domain_id,))
        self.db.execute("DELETE FROM domains WHERE id = ?", (domain_id,))

        self.db.commit()
        # Clear in-memory embeddings for products in this domain
        if domain in self._product_id_maps:
            for internal_id in self._product_id_maps[domain].values():
                self._product_embeddings.pop(internal_id, None)
        self._domain_ids.pop(domain, None)
        self._product_id_maps.pop(domain, None)
        self._ingested_domains.discard(domain)
