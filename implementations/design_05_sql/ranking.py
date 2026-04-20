"""Ranking logic for Design 05: SQL-First / SQLite + FTS5.

Three-signal ranking:
  - Attribute match score (weight: 0.50)
  - FTS5 BM25 relevance score (weight: 0.35)
  - Review sentiment score (weight: 0.15)

Includes near-miss tolerance and explanation generation.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3

logger = logging.getLogger(__name__)

# Default weights for the four ranking signals
W_ATTRIBUTE = 0.40
W_FTS = 0.25
W_EMBEDDING = 0.25
W_SENTIMENT = 0.10


def compute_attribute_scores(
    db: sqlite3.Connection,
    product_id: int,
    domain_id: int,
    filters: list[dict],
) -> dict[str, float]:
    """Compute per-attribute match scores for a product against filters.

    Returns a dict of attribute_name -> score (0.0 to 1.0).
    Near-misses get partial scores via linear decay.
    """
    if not filters:
        return {}

    scores = {}

    for f in filters:
        attr_name = f["attribute"]
        op = f["op"]
        target = f["value"]

        # Get the attribute definition
        attr_def = db.execute(
            "SELECT id, data_type, scale_min, scale_max FROM attribute_defs "
            "WHERE domain_id = ? AND name = ?",
            (domain_id, attr_name),
        ).fetchone()

        if attr_def is None:
            # Unknown attribute, skip
            continue

        attr_def_id, data_type, scale_min, scale_max = attr_def

        # Get the product's attribute value
        pa_row = db.execute(
            "SELECT value_numeric, value_text FROM product_attributes "
            "WHERE product_id = ? AND attribute_def_id = ?",
            (product_id, attr_def_id),
        ).fetchone()

        if pa_row is None:
            scores[attr_name] = 0.0
            continue

        value_numeric, value_text = pa_row

        if op in ("contains", "not_contains"):
            scores[attr_name] = _score_categorical(value_text, target, op)
        elif op == "eq":
            if data_type in ("numeric", "scale") and value_numeric is not None:
                try:
                    target_val = float(target)
                    scores[attr_name] = 1.0 if abs(value_numeric - target_val) < 0.01 else 0.0
                except (TypeError, ValueError):
                    scores[attr_name] = 0.0
            else:
                scores[attr_name] = 1.0 if str(value_text) == str(target) else 0.0
        elif op in ("gte", "lte"):
            scores[attr_name] = _score_numeric_with_near_miss(
                value_numeric, value_text, target, op, data_type, scale_min, scale_max, attr_name
            )
        else:
            scores[attr_name] = 0.0

    return scores


def _score_categorical(value_text: str | None, target, op: str) -> float:
    """Score a categorical attribute match."""
    if value_text is None:
        return 0.0 if op == "contains" else 1.0

    # Parse the stored value (could be JSON array or plain string)
    try:
        stored_values = json.loads(value_text)
        if isinstance(stored_values, str):
            stored_values = [stored_values]
    except (json.JSONDecodeError, TypeError):
        stored_values = [str(value_text)]

    # Normalize for comparison
    stored_lower = [str(v).lower().strip() for v in stored_values]
    target_lower = str(target).lower().strip()

    if op == "contains":
        # Check if target is in stored values (partial match allowed)
        for sv in stored_lower:
            if target_lower in sv or sv in target_lower:
                return 1.0
        return 0.0
    else:  # not_contains
        for sv in stored_lower:
            if target_lower in sv or sv in target_lower:
                return 0.0
        return 1.0


def _score_numeric_with_near_miss(
    value_numeric: float | None,
    value_text: str | None,
    target,
    op: str,
    data_type: str,
    scale_min: float | None,
    scale_max: float | None,
    attr_name: str,
) -> float:
    """Score a numeric attribute with near-miss tolerance.

    E.g., a 178cm ski still scores partially for a "180cm+" query.
    """
    # Special handling for lengths_available
    if attr_name == "lengths_available" and value_text:
        return _score_lengths_available(value_text, target, op)

    if value_numeric is None:
        return 0.0

    try:
        target_val = float(target)
    except (TypeError, ValueError):
        return 0.0

    actual = value_numeric

    # Check exact match
    if op == "gte" and actual >= target_val:
        return 1.0
    if op == "lte" and actual <= target_val:
        return 1.0

    # Near-miss: compute linear decay
    distance = abs(actual - target_val)

    # Determine the scale range for normalization
    if scale_min is not None and scale_max is not None:
        scale_range = scale_max - scale_min
    else:
        # For unbounded numeric, use the target as scale reference
        scale_range = max(abs(target_val) * 0.5, 10.0)

    if scale_range < 0.01:
        scale_range = 10.0

    normalized_distance = distance / scale_range
    score = max(0.0, 1.0 - normalized_distance)
    return score


def _score_lengths_available(value_text: str, target, op: str) -> float:
    """Score whether any available length matches a length requirement."""
    try:
        lengths = json.loads(value_text)
        if not isinstance(lengths, list):
            return 0.0
        target_val = float(target)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0

    if op == "gte":
        # Check if any available length >= target
        matching = [l for l in lengths if l >= target_val]
        if matching:
            return 1.0
        # Near-miss: closest length
        if lengths:
            closest = max(lengths)
            distance = target_val - closest
            if distance > 0:
                score = max(0.0, 1.0 - distance / 20.0)  # 20cm tolerance range
                return score
        return 0.0
    elif op == "lte":
        matching = [l for l in lengths if l <= target_val]
        if matching:
            return 1.0
        if lengths:
            closest = min(lengths)
            distance = closest - target_val
            if distance > 0:
                score = max(0.0, 1.0 - distance / 20.0)
                return score
        return 0.0

    return 0.0


def compute_embedding_scores(
    query_embedding: list[float],
    product_embeddings: dict[int, list[float]],
    product_ids: list[int],
) -> dict[int, float]:
    """Compute cosine similarity between query embedding and product embeddings.

    Returns a dict of product_id -> similarity score (0.0 to 1.0).
    """
    if not query_embedding or not product_embeddings:
        return {pid: 0.0 for pid in product_ids}

    # Precompute query norm
    q_norm = math.sqrt(sum(x * x for x in query_embedding))
    if q_norm < 1e-9:
        return {pid: 0.0 for pid in product_ids}

    scores = {}
    for pid in product_ids:
        emb = product_embeddings.get(pid)
        if emb is None:
            scores[pid] = 0.0
            continue
        dot = sum(a * b for a, b in zip(query_embedding, emb))
        p_norm = math.sqrt(sum(x * x for x in emb))
        if p_norm < 1e-9:
            scores[pid] = 0.0
        else:
            # Cosine similarity in [-1, 1], shift to [0, 1]
            cos_sim = dot / (q_norm * p_norm)
            scores[pid] = max(0.0, (cos_sim + 1.0) / 2.0)

    return scores


def compute_fts_scores(
    db: sqlite3.Connection,
    product_ids: list[int],
    keywords: str,
) -> dict[int, float]:
    """Compute FTS5 BM25 scores for products matching keywords.

    Returns a dict of product_id -> normalized score (0.0 to 1.0).
    """
    if not keywords or not keywords.strip() or not product_ids:
        return {pid: 0.0 for pid in product_ids}

    # Sanitize keywords for FTS5 - handle special characters
    sanitized = _sanitize_fts_query(keywords)
    if not sanitized:
        return {pid: 0.0 for pid in product_ids}

    try:
        # Query FTS5 for matching reviews, get BM25 scores per product
        placeholders = ",".join("?" * len(product_ids))
        rows = db.execute(
            f"""
            SELECT r.product_id, MIN(fts.rank) as best_rank
            FROM reviews_fts fts
            JOIN reviews r ON r.id = fts.rowid
            WHERE reviews_fts MATCH ?
              AND r.product_id IN ({placeholders})
            GROUP BY r.product_id
            """,
            [sanitized] + product_ids,
        ).fetchall()

        if not rows:
            return {pid: 0.0 for pid in product_ids}

        # FTS5 rank is negative (more negative = better match)
        raw_scores = {row[0]: row[1] for row in rows}
        return _normalize_bm25_scores(raw_scores, product_ids)

    except sqlite3.OperationalError as e:
        logger.warning("FTS query failed: %s (query: %s)", e, sanitized)
        return {pid: 0.0 for pid in product_ids}


def _sanitize_fts_query(keywords: str) -> str:
    """Sanitize a query string for FTS5 MATCH.

    Handles synonym-expanded queries with OR groups by converting them
    into individual quoted terms joined with OR, which FTS5 accepts.
    """
    if not keywords:
        return ""

    # If it contains OR groups from synonym expansion, we need to handle them
    # FTS5 requires proper syntax: terms separated by OR or implicit AND
    if "OR" in keywords and "(" in keywords:
        # Extract all quoted terms and bare words from the expanded query
        import re
        # Find all quoted strings and bare words
        all_terms = set()
        # Get quoted terms
        for match in re.finditer(r'"([^"]+)"', keywords):
            term = match.group(1).strip()
            if term:
                all_terms.add(term)
        # Get bare words outside of parenthesized groups
        remaining = re.sub(r'\([^)]*\)', '', keywords)
        for word in remaining.split():
            word = word.strip('"(),.:;!?').strip()
            if word and word != 'OR' and len(word) > 1:
                all_terms.add(word)

        if not all_terms:
            return ""

        # Build a simple OR query with all terms
        parts = []
        for term in sorted(all_terms):
            # Quote terms with spaces, hyphens, or other special chars
            if ' ' in term or '-' in term:
                parts.append(f'"{term}"')
            else:
                parts.append(term)
        return " OR ".join(parts)

    # Simple case: no synonym expansion
    # Split into tokens and quote any with special characters
    tokens = keywords.split()
    parts = []
    for token in tokens:
        # Remove non-alphanumeric except hyphens
        cleaned = ''.join(
            ch for ch in token if ch.isalnum() or ch in ('-', '_')
        )
        if not cleaned or len(cleaned) < 2:
            continue
        if '-' in cleaned:
            parts.append(f'"{cleaned}"')
        else:
            parts.append(cleaned)

    return ' '.join(parts)


def _normalize_bm25_scores(
    raw_scores: dict[int, float],
    all_product_ids: list[int],
) -> dict[int, float]:
    """Normalize FTS5 BM25 scores to 0-1 range.

    FTS5 bm25()/rank returns negative values where more negative = better match.
    We negate first so higher = better, then apply min-max normalization.
    """
    if not raw_scores:
        return {pid: 0.0 for pid in all_product_ids}

    negated = {pid: -score for pid, score in raw_scores.items()}
    values = list(negated.values())
    lo, hi = min(values), max(values)

    result = {}
    for pid in all_product_ids:
        if pid not in negated:
            result[pid] = 0.0
        elif hi - lo < 1e-9:
            # Single-value or identical scores: use sigmoid fallback
            result[pid] = 1.0 / (1.0 + math.exp(-negated[pid]))
        else:
            result[pid] = (negated[pid] - lo) / (hi - lo)

    return result


def compute_sentiment_scores(
    db: sqlite3.Connection,
    product_ids: list[int],
) -> dict[int, float]:
    """Compute average sentiment scores per product, normalized to 0-1.

    Sentiment is stored per-review as -1.0 to 1.0.
    We shift to 0-1 range: (sentiment + 1) / 2.
    """
    if not product_ids:
        return {}

    placeholders = ",".join("?" * len(product_ids))
    rows = db.execute(
        f"""
        SELECT product_id, AVG(sentiment) as avg_sentiment
        FROM reviews
        WHERE product_id IN ({placeholders})
        GROUP BY product_id
        """,
        product_ids,
    ).fetchall()

    sentiment_map = {row[0]: row[1] for row in rows}

    result = {}
    for pid in product_ids:
        raw = sentiment_map.get(pid, 0.0)
        # Shift from [-1, 1] to [0, 1]
        result[pid] = (raw + 1.0) / 2.0

    return result


def build_explanation(
    attr_scores: dict[str, float],
    fts_score: float,
    embed_score: float,
    sent_score: float,
    total_score: float,
    filters: list[dict],
) -> str:
    """Build a human-readable explanation with score breakdowns.

    Includes near-miss annotations for partial attribute matches.
    """
    lines = [f"  Score: {total_score:.3f}", "  Attribute breakdown:"]

    if attr_scores:
        max_name_len = max(len(name) for name in attr_scores) if attr_scores else 0
        for attr_name, score in attr_scores.items():
            padded = attr_name.ljust(max_name_len)
            if score >= 1.0:
                annotation = "(exact match)"
            elif score > 0:
                penalty = score - 1.0
                # Find the corresponding filter for context
                target_info = ""
                for f in filters:
                    if f["attribute"] == attr_name:
                        target_info = f" target={f['op']}{f['value']}"
                        break
                annotation = f"[NEAR-MISS:{target_info} penalty={penalty:+.2f}]"
            else:
                annotation = "(no match)"
            lines.append(f"    {padded}  {score:.2f}  {annotation}")
        avg_attr = sum(attr_scores.values()) / len(attr_scores)
    else:
        lines.append("    (no attribute filters)")
        avg_attr = 0.0

    lines.append(f"  FTS relevance:  {fts_score:.2f}")
    lines.append(f"  Embedding sim:  {embed_score:.2f}")
    lines.append(f"  Sentiment:      {sent_score:.2f}")
    lines.append(
        f"  Weighted total: {W_ATTRIBUTE:.2f}*{avg_attr:.2f} + "
        f"{W_FTS:.2f}*{fts_score:.2f} + "
        f"{W_EMBEDDING:.2f}*{embed_score:.2f} + "
        f"{W_SENTIMENT:.2f}*{sent_score:.2f} = {total_score:.3f}"
    )
    return "\n".join(lines)
