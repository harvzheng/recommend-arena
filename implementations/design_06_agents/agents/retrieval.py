"""Retrieval Agent -- fetches candidate products via filters + vector similarity.

This agent is fully deterministic (no LLM). It applies hard filters from the
parsed query via SQLite, then fetches similar products from ChromaDB, and
merges the two result sets.
"""

from __future__ import annotations

import logging

from ..state import ParsedQuery, ProductAttributes

logger = logging.getLogger(__name__)


def retrieval_agent(state: dict, store=None) -> dict:
    """Retrieve candidate products matching the parsed query.

    Uses two retrieval strategies:
    1. Hard filters via SQLite (must-match constraints)
    2. Soft preferences via ChromaDB vector similarity

    The union of both result sets forms the candidate pool.

    Args:
        state: Pipeline state with "parsed_query" and "domain".
        store: Store instance for data access.

    Returns:
        Updated state with "candidates" set.
    """
    if store is None:
        return {**state, "candidates": [], "errors": state.get("errors", []) + ["No store provided"]}

    parsed: ParsedQuery | None = state.get("parsed_query")
    domain = state.get("domain", "ski")

    if parsed is None:
        # No parsed query -- return all products
        candidates = store.query_by_domain(domain)
        return {**state, "candidates": candidates}

    # Strategy 1: Hard filter via SQLite
    filtered = store.query_by_filters(domain, parsed.hard_filters)
    filtered_ids = {p.product_id for p in filtered}

    # Strategy 2: Vector similarity via ChromaDB
    vector_query = _build_vector_query(parsed)
    if vector_query:
        vector_ids = store.vector_search(domain, vector_query, n_results=25)
    else:
        vector_ids = []

    # Merge: union of both sets, but prioritize filter matches
    all_ids = list(filtered_ids)
    for vid in vector_ids:
        if vid not in filtered_ids:
            all_ids.append(vid)

    # If hard filters returned results, only use those (they're must-match)
    if parsed.hard_filters and filtered:
        candidates = filtered
    elif filtered:
        # Have filters but also want vector results
        vector_products = store.get_products_by_ids(
            [vid for vid in vector_ids if vid not in filtered_ids], domain
        )
        candidates = filtered + vector_products
    else:
        # No filter matches or no filters -- use vector results + all products
        if vector_ids:
            candidates = store.get_products_by_ids(vector_ids, domain)
        else:
            candidates = store.query_by_domain(domain)

    # Ensure we have at least some candidates even with strict filters
    if not candidates:
        logger.info(
            "No candidates after filtering, falling back to all products"
        )
        candidates = store.query_by_domain(domain)

    logger.info(
        "Retrieved %d candidates (filtered=%d, vector=%d)",
        len(candidates),
        len(filtered),
        len(vector_ids),
    )

    return {**state, "candidates": candidates}


def _build_vector_query(parsed: ParsedQuery) -> str:
    """Build a text query for vector search from parsed preferences."""
    parts = []

    if parsed.freetext_intent:
        parts.append(parsed.freetext_intent)

    for attr, val in parsed.soft_preferences.items():
        if val >= 7:
            parts.append(f"high {attr}")
        elif val <= 3:
            parts.append(f"low {attr}")

    for attr, val in parsed.negative_preferences.items():
        if val >= 7:
            parts.append(f"not {attr}")

    return " ".join(parts)
