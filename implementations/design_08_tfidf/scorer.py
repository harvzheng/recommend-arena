"""Weighted cosine similarity scoring and explanation generation.

Scores query vectors against product vectors using IDF-weighted cosine
similarity.  Scores are clamped to [0, 1].
"""

from __future__ import annotations

import math


def weighted_cosine(
    q: dict[str, float],
    p: dict[str, float],
    w: dict[str, float],
) -> float:
    """Compute IDF-weighted cosine similarity between *q* and *p*.

    Args:
        q: Query feature vector (may contain negative values for negations).
        p: Product feature vector.
        w: Weight vector (typically IDF weights).

    Returns:
        Similarity score clamped to [0, 1].
    """
    all_keys = set(q.keys()) | set(p.keys())
    if not all_keys:
        return 0.0

    dot = 0.0
    q_norm_sq = 0.0
    p_norm_sq = 0.0

    for k in all_keys:
        qk = q.get(k, 0.0)
        pk = p.get(k, 0.0)
        wk = w.get(k, 1.0)

        dot += qk * pk * wk
        q_norm_sq += (qk * wk) ** 2
        p_norm_sq += (pk * wk) ** 2

    q_norm = math.sqrt(q_norm_sq)
    p_norm = math.sqrt(p_norm_sq)

    if q_norm == 0.0 or p_norm == 0.0:
        return 0.0

    raw = dot / (q_norm * p_norm)
    return max(0.0, min(1.0, raw))


def per_attribute_contributions(
    q: dict[str, float],
    p: dict[str, float],
    w: dict[str, float],
    min_abs: float = 0.005,
) -> dict[str, float]:
    """Return per-key contribution to the dot product.

    Only keys whose absolute contribution exceeds *min_abs* are included.
    """
    contribs: dict[str, float] = {}
    for k in set(q.keys()) & set(p.keys()):
        c = q[k] * p[k] * w.get(k, 1.0)
        if abs(c) > min_abs:
            contribs[k] = round(c, 4)
    return contribs


def explain_score(
    q: dict[str, float],
    p: dict[str, float],
    w: dict[str, float],
) -> str:
    """Generate a human-readable explanation of how a score was computed.

    Returns a string describing positive matches and negative penalties.
    """
    positives: list[tuple[str, float, str]] = []
    negatives: list[tuple[str, float]] = []

    for k in set(q.keys()) & set(p.keys()):
        contrib = q[k] * p[k] * w.get(k, 1.0)
        if contrib > 0.01:
            if contrib > 0.3:
                strength = "high"
            elif contrib > 0.15:
                strength = "moderate"
            else:
                strength = "slight"
            positives.append((k, contrib, strength))
        elif contrib < -0.01:
            negatives.append((k, contrib))

    positives.sort(key=lambda x: x[1], reverse=True)
    negatives.sort(key=lambda x: x[1])

    parts: list[str] = []
    if positives:
        matched = ", ".join(
            f"{k} ({strength} match, {c:+.2f})"
            for k, c, strength in positives
        )
        parts.append(f"Matched on {matched}")
    if negatives:
        penalized = ", ".join(f"{k} ({c:+.2f})" for k, c in negatives)
        parts.append(f"Penalized for {penalized}")

    return ". ".join(parts) + "." if parts else "No significant attribute overlap."
