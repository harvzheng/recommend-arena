"""Design #9: Faceted Search Recommendation System.

Uses SQLite FTS5 as the search backend with structured facet tables.
LLM is used for facet extraction (ingestion) and query parsing (search).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# Add project root to path for shared imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .aggregation import aggregate_product_facets, store_extracted_facets
from .extraction import extract_facets
from .indexer import FacetedIndex
from .query_parser import parse_query
from .searcher import execute_search

logger = logging.getLogger(__name__)

# Directory containing domain YAML configs
_DOMAINS_DIR = Path(__file__).parent / "domains"


class FacetedSearchRecommender:
    """Recommendation system based on faceted search with SQLite FTS5.

    Implements the shared Recommender protocol.
    """

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm or get_provider()
        self.index = FacetedIndex()
        self._domain_configs: dict[str, dict] = {}
        self._collection_names: dict[str, str] = {}
        self._ingested_domains: set[str] = set()
        # Cache for facet extraction to avoid redundant LLM calls
        self._extraction_cache: dict[str, dict] = {}

    def ingest(
        self,
        products: list[dict],
        reviews: list[dict],
        domain: str,
    ) -> None:
        """Ingest product and review data for a given domain.

        Steps:
            1. Load domain config (YAML)
            2. Create the search index collection
            3. Extract facets from reviews using LLM (with caching)
            4. Aggregate facets per product
            5. Build and index documents
        """
        # Load domain configuration
        domain_config = self._load_domain_config(domain)
        collection_name = domain_config.get("collection", domain)
        self._collection_names[domain] = collection_name

        # Create collection in the index
        self.index.create_collection(collection_name, domain_config)

        # Build a lookup for reviews by product
        reviews_by_product: dict[str, list[dict]] = {}
        for review in reviews:
            pid = review.get("product_id", "")
            reviews_by_product.setdefault(pid, []).append(review)

        # Use an in-memory SQLite DB for facet staging
        import sqlite3
        staging_db = sqlite3.connect(":memory:")

        # Step 1-2: Extract facets from reviews
        for review in reviews:
            review_id = review.get("review_id") or review.get("reviewer", "")
            product_id = review.get("product_id", "")
            review_text = review.get("review_text") or review.get("text", "")

            if not review_text:
                continue

            # Check cache
            cache_key = self._cache_key(review_text, domain)
            if cache_key in self._extraction_cache:
                extracted = self._extraction_cache[cache_key]
            else:
                extracted = extract_facets(review_text, domain_config, self.llm)
                self._extraction_cache[cache_key] = extracted

            store_extracted_facets(staging_db, review_id, product_id, extracted)

        # Step 3-4: Aggregate and build documents for each product
        for product in products:
            pid = product.get("id") or product.get("product_id", "")
            product_reviews = reviews_by_product.get(pid, [])

            # Aggregate LLM-extracted facets
            aggregated = aggregate_product_facets(staging_db, pid, domain_config)

            # Build the indexable document
            doc = self._build_document(
                product, aggregated, product_reviews, domain, domain_config
            )

            # Index it
            self.index.upsert_document(collection_name, doc)

        staging_db.close()
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
        """Query the recommendation system with natural language."""
        domain_config = self._load_domain_config(domain)
        collection_name = self._collection_names.get(
            domain, domain_config.get("collection", domain)
        )

        # Parse the natural language query into structured search params
        parsed = parse_query(query_text, domain_config, self.llm)
        logger.debug("Parsed query: %s", parsed)

        # Execute the faceted search
        results = execute_search(
            self.index,
            collection_name,
            parsed,
            domain_config,
            top_k=top_k,
        )

        return results

    def _load_domain_config(self, domain: str) -> dict:
        """Load and cache domain configuration from YAML."""
        if domain in self._domain_configs:
            return self._domain_configs[domain]

        yaml_path = _DOMAINS_DIR / f"{domain}.yaml"
        if not yaml_path.exists():
            # Generate a minimal config from data
            logger.warning(
                "No YAML config for domain '%s'; using generic config", domain
            )
            config = self._generate_generic_config(domain)
        else:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)

        self._domain_configs[domain] = config
        return config

    def _generate_generic_config(self, domain: str) -> dict:
        """Generate a minimal domain config when no YAML exists."""
        return {
            "domain": domain,
            "collection": domain,
            "facets": {
                "quality": {
                    "type": "numeric",
                    "range": [0.0, 1.0],
                    "description": "Overall quality rating",
                },
            },
            "ranking_defaults": {
                "sort_by": "popularity:desc",
                "primary_facets": ["quality"],
            },
        }

    def _build_document(
        self,
        product: dict,
        aggregated_facets: dict,
        reviews: list[dict],
        domain: str,
        domain_config: dict,
    ) -> dict:
        """Build a document for indexing from product data + aggregated facets."""
        pid = product.get("id") or product.get("product_id", "")
        specs = product.get("specs", {})
        attrs = product.get("attributes", {})
        metadata = product.get("metadata", {})

        doc: dict = {
            "id": pid,
            "name": product.get("name") or product.get("product_name", ""),
            "brand": product.get("brand", ""),
            "domain": domain,
            "category": product.get("category", ""),
            "review_count": len(reviews),
            "popularity": len(reviews),  # Use review count as popularity proxy
        }

        facets = domain_config["facets"]

        # Merge product attributes with LLM-extracted facets
        # Product attributes serve as ground truth, LLM-extracted augments
        for fname, fdef in facets.items():
            ftype = fdef["type"]

            if ftype == "numeric":
                # Check product attributes first (scale from 0-10 to 0-1)
                attr_val = attrs.get(fname)
                if attr_val is not None:
                    try:
                        doc[fname] = float(attr_val) / 10.0
                    except (ValueError, TypeError):
                        pass
                # Also check for renamed attributes
                attr_mappings = self._get_attr_mappings(domain)
                for attr_name, facet_name in attr_mappings.items():
                    if facet_name == fname and attr_name in attrs:
                        try:
                            doc[fname] = float(attrs[attr_name]) / 10.0
                        except (ValueError, TypeError):
                            pass
                # LLM-extracted facets as fallback or supplement
                if fname not in doc and fname in aggregated_facets:
                    doc[fname] = aggregated_facets[fname]

            elif ftype == "categorical":
                # Product terrain/category first, then check specs
                cat_val = attrs.get(fname, [])
                if not cat_val:
                    cat_val = specs.get(fname, [])
                if not cat_val:
                    cat_val = metadata.get(fname, [])
                if isinstance(cat_val, str):
                    cat_val = [cat_val]
                if cat_val:
                    doc[fname] = cat_val
                elif fname in aggregated_facets:
                    doc[fname] = aggregated_facets[fname]

            elif ftype == "boolean":
                if fname in aggregated_facets:
                    doc[fname] = aggregated_facets[fname]
                elif fname == "has_rocker":
                    rocker_profile = specs.get("rocker_profile", "")
                    doc[fname] = "rocker" in rocker_profile.lower()
                elif fname == "has_plate":
                    binding = specs.get("binding_system", "")
                    doc[fname] = binding == "plate"
                elif fname == "has_carbon_plate":
                    category = product.get("category", "")
                    doc[fname] = "carbon" in category.lower()

            elif ftype == "spec_numeric":
                # Map from product specs
                spec_mappings = {
                    "waist_width_mm": "waist_width_mm",
                    "turn_radius_m": "turn_radius_m",
                    "weight_g": "weight_g_per_ski",
                    "length_cm": None,  # handled specially
                    "heel_drop_mm": "heel_drop_mm",
                    "stack_height_mm": "stack_height_mm",
                }
                spec_key = spec_mappings.get(fname, fname)

                # Check specs dict and metadata
                for source in [specs, metadata]:
                    if spec_key and spec_key in source:
                        try:
                            doc[fname] = float(source[spec_key])
                        except (ValueError, TypeError):
                            pass
                    # Also try the facet name directly
                    if fname in source:
                        try:
                            doc[fname] = float(source[fname])
                        except (ValueError, TypeError):
                            pass

                # For weight_g, also check weight_g directly
                if fname == "weight_g":
                    for key in ["weight_g", "weight_g_per_ski"]:
                        for source in [specs, metadata]:
                            if key in source:
                                try:
                                    doc[fname] = float(source[key])
                                except (ValueError, TypeError):
                                    pass

        # Store available lengths for range queries
        lengths = specs.get("lengths_cm", [])
        if lengths:
            doc["lengths_cm"] = [float(l) for l in lengths]
            # Also set length_cm as max length for filtering
            doc["length_cm"] = max(float(l) for l in lengths)

        # Build review summary for text search
        review_texts = []
        for r in reviews:
            text = r.get("review_text") or r.get("text", "")
            if text:
                review_texts.append(text)

        if review_texts:
            # Use a condensed summary from review texts
            doc["review_summary"] = self._build_review_summary(
                review_texts, product.get("name", "")
            )
        else:
            # Use product metadata as summary
            parts = [product.get("name", ""), product.get("category", "")]
            if attrs.get("terrain"):
                terrain = attrs["terrain"]
                if isinstance(terrain, list):
                    parts.extend(terrain)
                else:
                    parts.append(str(terrain))
            doc["review_summary"] = " ".join(parts)

        return doc

    def _build_review_summary(self, review_texts: list[str], product_name: str) -> str:
        """Build a searchable summary from review texts.

        Instead of an LLM call (expensive during ingestion), extract key
        phrases and create a keyword-rich summary.
        """
        # Concatenate and truncate reviews for a keyword-rich summary
        combined = " ".join(review_texts)
        # Limit to a reasonable length for FTS indexing
        if len(combined) > 2000:
            combined = combined[:2000]
        return f"{product_name}. {combined}"

    def _get_attr_mappings(self, domain: str) -> dict[str, str]:
        """Map product attribute names to facet names."""
        if domain == "ski":
            return {
                "damp": "dampness",
                "stability_at_speed": "stability",
                "powder_float": "powder_float",
            }
        elif domain == "running_shoe":
            return {}
        return {}

    @staticmethod
    def _cache_key(text: str, domain: str) -> str:
        """Create a cache key for review text."""
        h = hashlib.md5(f"{domain}:{text}".encode()).hexdigest()
        return h
