"""Faceted search execution + ranking + result formatting."""

from __future__ import annotations

import logging
import sys
import os

# Add project root to path for shared imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.interface import RecommendationResult

from .indexer import FacetedIndex

logger = logging.getLogger(__name__)


def execute_search(
    index: FacetedIndex,
    collection_name: str,
    parsed_query: dict,
    domain_config: dict,
    top_k: int = 10,
) -> list[RecommendationResult]:
    """Execute a faceted search and return formatted results.

    Args:
        index: FacetedIndex instance.
        collection_name: Collection to search.
        parsed_query: Output from query_parser.parse_query().
        domain_config: Domain configuration.
        top_k: Number of results to return.

    Returns:
        List of RecommendationResult, sorted by score descending.
    """
    text_query = parsed_query.get("text_query", "*")
    filters = parsed_query.get("filters", {})
    sort_fields = parsed_query.get("sort_fields", [])

    # Execute the search with generous limit for re-ranking
    hits = index.search(
        collection_name=collection_name,
        text_query=text_query,
        filters=filters,
        sort_fields=sort_fields,
        limit=top_k * 3,
    )

    if not hits:
        # If filtered search returns nothing, try text-only
        if filters:
            logger.info("Filtered search returned 0 results; retrying text-only")
            hits = index.search(
                collection_name=collection_name,
                text_query=parsed_query.get("text_query", "*")
                if parsed_query.get("text_query", "*") != "*"
                else text_query,
                filters={},
                sort_fields=[],
                limit=top_k * 3,
            )
        if not hits:
            # Last resort: return all products
            hits = index.search(
                collection_name=collection_name,
                text_query=None,
                filters={},
                sort_fields=[],
                limit=top_k * 3,
            )

    # Score and rank results
    scored_hits = _score_results(hits, parsed_query, domain_config)

    # Sort by score descending
    scored_hits.sort(key=lambda x: x[1], reverse=True)

    # Format into RecommendationResult objects
    results = []
    for doc, score, matched_attrs in scored_hits[:top_k]:
        explanation = _build_explanation(doc, matched_attrs, parsed_query)
        results.append(
            RecommendationResult(
                product_id=doc.get("id", ""),
                product_name=doc.get("name", ""),
                score=score,
                explanation=explanation,
                matched_attributes=matched_attrs,
            )
        )

    return results


def _score_results(
    hits: list[dict],
    parsed_query: dict,
    domain_config: dict,
) -> list[tuple[dict, float, dict]]:
    """Score each hit based on filter match quality and text relevance.

    Returns:
        List of (doc, score, matched_attributes) tuples.
    """
    filters = parsed_query.get("filters", {})
    sort_fields = parsed_query.get("sort_fields", [])
    text_query = parsed_query.get("text_query", "*")
    has_text = text_query and text_query.strip() != "*"
    facets = domain_config["facets"]

    # Compute text relevance scores (normalized)
    if has_text and hits:
        text_ranks = [abs(h.get("_text_rank", 0)) for h in hits]
        max_rank = max(text_ranks) if text_ranks else 1
        if max_rank == 0:
            max_rank = 1
    else:
        text_ranks = [0.0] * len(hits)
        max_rank = 1

    results = []
    for i, doc in enumerate(hits):
        matched_attrs: dict[str, float] = {}
        score_components: list[float] = []

        # Text relevance component
        if has_text:
            # FTS5 rank: lower is better match, so invert
            raw_rank = abs(doc.get("_text_rank", 0))
            if max_rank > 0:
                text_score = 1.0 - (raw_rank / max_rank) if raw_rank > 0 else 1.0
            else:
                text_score = 0.5
            score_components.append(text_score * 0.3)
        else:
            score_components.append(0.15)  # Neutral text component

        # Facet filter match components
        if filters:
            filter_scores = []
            for field, (op, target) in filters.items():
                fdef = facets.get(field, {})
                ftype = fdef.get("type", "")
                val = doc.get(field)

                if val is None:
                    filter_scores.append(0.0)
                    matched_attrs[field] = 0.0
                    continue

                match_quality = _compute_match_quality(val, op, target, ftype)
                filter_scores.append(match_quality)
                matched_attrs[field] = match_quality

            if filter_scores:
                avg_filter = sum(filter_scores) / len(filter_scores)
                score_components.append(avg_filter * 0.6)
            else:
                score_components.append(0.3)
        else:
            score_components.append(0.3)

        # Sort field bonus: boost documents that rank well on sort criteria
        if sort_fields:
            sort_score = _compute_sort_score(doc, sort_fields, facets)
            score_components.append(sort_score * 0.1)
        else:
            score_components.append(0.05)

        total_score = min(1.0, sum(score_components))
        results.append((doc, total_score, matched_attrs))

    return results


def _compute_match_quality(
    val, op: str, target: str, ftype: str
) -> float:
    """Compute how well a value satisfies a filter condition (0-1)."""
    if ftype in ("numeric", "spec_numeric"):
        try:
            target_f = float(target)
            val_f = float(val)
        except (ValueError, TypeError):
            return 0.0

        if op in (">", ">="):
            if target_f == 0:
                return 1.0
            ratio = val_f / target_f
            return min(1.0, max(0.0, ratio))
        elif op in ("<", "<="):
            if val_f == 0:
                return 1.0
            ratio = target_f / val_f
            return min(1.0, max(0.0, ratio))
        else:
            # Exact match: score by closeness
            if target_f == 0:
                return 1.0 if val_f == 0 else 0.0
            diff = abs(val_f - target_f) / max(abs(target_f), 1.0)
            return max(0.0, 1.0 - diff)

    elif ftype == "categorical":
        if isinstance(val, list):
            return 1.0 if target in val else 0.0
        return 1.0 if val == target else 0.0

    elif ftype == "boolean":
        val_bool = str(val).lower() in ("1", "true", "yes")
        target_bool = target.lower() in ("true", "1", "yes")
        return 1.0 if val_bool == target_bool else 0.0

    return 0.5


def _compute_sort_score(
    doc: dict,
    sort_fields: list[tuple[str, str]],
    facets: dict,
) -> float:
    """Score based on sort field values (higher is better for desc, lower for asc)."""
    scores = []
    for field, direction in sort_fields:
        val = doc.get(field)
        if val is None:
            scores.append(0.0)
            continue
        try:
            val_f = float(val)
            # For numeric facets in 0-1 range
            fdef = facets.get(field, {})
            if fdef.get("type") == "numeric":
                score = val_f if direction == "desc" else (1.0 - val_f)
            else:
                score = 0.5  # Can't normalize arbitrary spec values
            scores.append(score)
        except (ValueError, TypeError):
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.5


def _build_explanation(
    doc: dict,
    matched_attrs: dict[str, float],
    parsed_query: dict,
) -> str:
    """Build a human-readable explanation of why this product matched."""
    parts = []

    for field, quality in sorted(matched_attrs.items(), key=lambda x: -x[1]):
        actual = doc.get(field)
        if actual is None:
            continue

        if isinstance(actual, float):
            parts.append(f"{field}={actual:.2f} (match: {quality:.0%})")
        elif isinstance(actual, list):
            parts.append(f"{field}={', '.join(str(v) for v in actual)} (match: {quality:.0%})")
        else:
            parts.append(f"{field}={actual} (match: {quality:.0%})")

    if parts:
        return "; ".join(parts)

    # Fallback to review summary excerpt
    summary = doc.get("review_summary", "")
    if summary:
        return summary[:150]
    return doc.get("name", "Unknown product")
