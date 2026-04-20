"""Match scoring from posterior distributions.

Computes how well a product's posteriors match a set of query constraints.
Supports three ranking modes: expected value, UCB (optimistic), LCB (conservative).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm as sp_norm

from .beliefs import ProductBelief
from .schema import AttributeSpec, get_spec


# ---------------------------------------------------------------------------
# Constraint data structure
# ---------------------------------------------------------------------------

@dataclass
class AttributeConstraint:
    attribute: str
    value: str
    strength: float = 1.0


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_product(
    belief: ProductBelief,
    constraints: list[AttributeConstraint],
    schema: list[AttributeSpec],
) -> tuple[float, float]:
    """Compute match score and uncertainty for one product against query constraints.

    Returns:
        (score, uncertainty) where score is the product of weighted per-attribute
        probabilities and uncertainty is propagated from posterior variances.
    """
    if not constraints:
        return 1.0, 0.0

    log_score = 0.0
    total_variance = 0.0

    for c in constraints:
        spec = get_spec(schema, c.attribute)
        if spec is None:
            continue
        posterior = belief.posteriors.get(c.attribute)
        if posterior is None:
            continue

        if spec.attr_type in ("ordinal", "categorical"):
            if spec.levels is None:
                continue
            alpha = posterior["alpha"]
            idx = _level_index(spec.levels, c.value)
            if idx is None:
                continue

            alpha_sum = sum(alpha)
            p = alpha[idx] / alpha_sum
            # Variance of the Dirichlet marginal (Beta distribution)
            var = (alpha[idx] * (alpha_sum - alpha[idx])) / (
                alpha_sum ** 2 * (alpha_sum + 1)
            )
            log_score += c.strength * math.log(p + 1e-10)
            total_variance += c.strength ** 2 * var

        elif spec.attr_type == "continuous":
            try:
                target = float(c.value)
            except (ValueError, TypeError):
                continue

            mu = posterior["mu"]
            sigma = posterior["sigma"]
            epsilon = spec.prior.get("epsilon", spec.prior.get("obs_sigma", 5.0))

            p = (
                sp_norm.cdf(target + epsilon, mu, sigma)
                - sp_norm.cdf(target - epsilon, mu, sigma)
            )
            log_score += c.strength * math.log(p + 1e-10)
            total_variance += c.strength ** 2 * sigma ** 2

    score = math.exp(log_score)
    uncertainty = math.sqrt(total_variance) if total_variance > 0 else 0.0
    return score, uncertainty


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_products(
    scored: list[tuple[str, float, float]],
    mode: str = "expected",
    k: float = 1.0,
) -> list[tuple[str, float, float]]:
    """Rank products by score with optional uncertainty adjustments.

    Args:
        scored: List of (product_id, score, uncertainty).
        mode: "expected" | "optimistic" (UCB) | "conservative" (LCB).
        k: Multiplier for uncertainty in UCB/LCB modes.
    """
    if mode == "optimistic":
        return sorted(scored, key=lambda x: x[1] + k * x[2], reverse=True)
    elif mode == "conservative":
        return sorted(scored, key=lambda x: x[1] - k * x[2], reverse=True)
    else:  # expected
        return sorted(scored, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_scores(
    scored: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Min-max normalize raw scores to [0, 1] across all candidates."""
    if not scored:
        return scored
    raw_scores = [s[1] for s in scored]
    s_min, s_max = min(raw_scores), max(raw_scores)
    if s_max - s_min < 1e-12:
        return [(pid, 1.0, unc) for pid, _, unc in scored]
    return [
        (pid, (raw - s_min) / (s_max - s_min), unc)
        for pid, raw, unc in scored
    ]


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------

def build_explanation(
    belief: ProductBelief,
    constraints: list[AttributeConstraint],
    schema: list[AttributeSpec],
) -> tuple[str, dict[str, float]]:
    """Build a human-readable explanation and per-attribute probability breakdown."""
    parts: list[str] = []
    matched_attributes: dict[str, float] = {}

    for c in constraints:
        spec = get_spec(schema, c.attribute)
        if spec is None:
            continue
        posterior = belief.posteriors.get(c.attribute)
        if posterior is None:
            continue

        if spec.attr_type in ("ordinal", "categorical"):
            if spec.levels is None:
                continue
            alpha = posterior["alpha"]
            idx = _level_index(spec.levels, c.value)
            if idx is None:
                continue
            alpha_sum = sum(alpha)
            p = alpha[idx] / alpha_sum
            # Observations = alpha[i] minus the prior (which was 1)
            obs_count = int(alpha[idx] - 1)
            total_obs = int(alpha_sum - len(alpha))
            parts.append(
                f'P({c.value}) = {p:.2f} (from {obs_count}/{total_obs} reviews)'
            )
            matched_attributes[c.attribute] = round(p, 4)

        elif spec.attr_type == "continuous":
            try:
                target = float(c.value)
            except (ValueError, TypeError):
                continue
            mu = posterior["mu"]
            sigma = posterior["sigma"]
            epsilon = spec.prior.get("epsilon", spec.prior.get("obs_sigma", 5.0))
            p = (
                sp_norm.cdf(target + epsilon, mu, sigma)
                - sp_norm.cdf(target - epsilon, mu, sigma)
            )
            parts.append(
                f'P({c.attribute} in [{target - epsilon:.0f}, {target + epsilon:.0f}]) '
                f'= {p:.2f} (posterior mu={mu:.1f}, sigma={sigma:.1f})'
            )
            matched_attributes[c.attribute] = round(float(p), 4)

    # Confidence label
    n = belief.evidence_count
    if n >= 50:
        confidence = "high"
    elif n >= 15:
        confidence = "medium"
    else:
        confidence = "low"

    if parts:
        explanation = (
            f"Recommended because {', '.join(parts)}. "
            f"Confidence: {confidence} ({n} total observations)."
        )
    else:
        explanation = f"General match. Confidence: {confidence} ({n} observations)."

    return explanation, matched_attributes


# ---------------------------------------------------------------------------
# Text search fallback for unmapped query terms
# ---------------------------------------------------------------------------

def text_search_fallback(
    unmapped_terms: list[str],
    product_ids: list[str],
    review_texts_by_product: dict[str, list[str]],
) -> dict[str, float]:
    """Score products by how often unmapped terms appear in their reviews."""
    if not unmapped_terms:
        return {pid: 0.0 for pid in product_ids}

    scores: dict[str, float] = {}
    for pid in product_ids:
        texts = review_texts_by_product.get(pid, [])
        corpus = " ".join(t.lower() for t in texts)
        term_hits = sum(1 for t in unmapped_terms if t.lower() in corpus)
        scores[pid] = term_hits / len(unmapped_terms)
    return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _level_index(levels: list[str], value: str) -> int | None:
    """Case-insensitive level lookup."""
    val_lower = value.lower().replace(" ", "_").replace("-", "_")
    for i, lvl in enumerate(levels):
        if lvl.lower().replace(" ", "_").replace("-", "_") == val_lower:
            return i
    return None
