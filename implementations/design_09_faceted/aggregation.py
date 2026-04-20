"""Per-product facet aggregation from multiple reviews."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict

logger = logging.getLogger(__name__)


def store_extracted_facets(
    db: sqlite3.Connection,
    review_id: str,
    product_id: str,
    facets: dict,
) -> None:
    """Store extracted facets from one review into the staging database."""
    db.execute(
        "CREATE TABLE IF NOT EXISTS extracted_facets ("
        "  review_id TEXT, product_id TEXT, facet_name TEXT, "
        "  facet_value TEXT, "
        "  PRIMARY KEY (review_id, facet_name))"
    )
    for name, value in facets.items():
        db.execute(
            "INSERT OR REPLACE INTO extracted_facets "
            "(review_id, product_id, facet_name, facet_value) VALUES (?, ?, ?, ?)",
            (review_id, product_id, name, json.dumps(value)),
        )
    db.commit()


def aggregate_product_facets(
    db: sqlite3.Connection,
    product_id: str,
    domain_config: dict,
) -> dict:
    """Aggregate facets for a single product across all its reviews.

    Returns:
        Dict mapping facet names to aggregated values.
    """
    rows = db.execute(
        "SELECT facet_name, facet_value FROM extracted_facets "
        "WHERE product_id = ? ORDER BY facet_name",
        (product_id,),
    ).fetchall()

    if not rows:
        return {}

    grouped: dict[str, list] = defaultdict(list)
    for name, raw_value in rows:
        grouped[name].append(json.loads(raw_value))

    facets = domain_config["facets"]
    aggregated = {}

    for name, values in grouped.items():
        if name not in facets:
            continue
        fdef = facets[name]
        ftype = fdef["type"]

        if ftype == "numeric":
            # Simple mean of numeric scores
            nums = [float(v) for v in values if _is_number(v)]
            if nums:
                aggregated[name] = sum(nums) / len(nums)
        elif ftype == "categorical":
            # Union of all mentioned categories
            cats: set[str] = set()
            for v in values:
                if isinstance(v, list):
                    cats.update(v)
                elif isinstance(v, str):
                    cats.add(v)
            aggregated[name] = sorted(cats)
        elif ftype == "boolean":
            # Majority vote
            bools = [bool(v) for v in values]
            aggregated[name] = sum(bools) > len(bools) / 2
        elif ftype == "spec_numeric":
            # Take the most common value or mean
            nums = [float(v) for v in values if _is_number(v)]
            if nums:
                aggregated[name] = sum(nums) / len(nums)

    return aggregated


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False
