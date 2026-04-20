"""LLM-based review extraction — turns raw reviews into structured profiles."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from .prompts import EXTRACTION_PROMPT

if TYPE_CHECKING:
    from shared.llm_provider import LLMProvider
    from .cache import JudgeCache

logger = logging.getLogger(__name__)


def extract_profile(
    product_id: str,
    product_name: str,
    domain: str,
    reviews: list[str],
    provider: "LLMProvider",
    cache: "JudgeCache",
) -> dict:
    """Extract a structured profile for a product from its reviews.

    Returns a dict with keys ``profile`` (the structured extraction) and
    ``summary`` (a natural-language summary used for embedding).
    """
    # Check cache first
    cached = cache.get_extraction(product_id)
    if cached is not None:
        return cached

    if not reviews:
        # No reviews — build a minimal profile from product name alone
        profile = {
            "attributes": {},
            "consensus_summary": f"{product_name} — no reviews available.",
            "reviewer_disagreements": "none",
            "sentiment": "neutral",
            "review_count": 0,
        }
        summary = f"{product_name}: no reviews available."
        cache.put_extraction(product_id, domain, profile, summary)
        return {"profile": profile, "summary": summary}

    reviews_text = "\n---\n".join(reviews)
    prompt = EXTRACTION_PROMPT.format(
        domain=domain,
        product_name=product_name,
        review_count=len(reviews),
        reviews_text=reviews_text,
    )

    try:
        raw = provider.generate(prompt, json_mode=True)
        profile = json.loads(raw)
    except (json.JSONDecodeError, RuntimeError) as exc:
        logger.warning(
            "Extraction failed for %s (%s): %s — using fallback",
            product_id,
            product_name,
            exc,
        )
        profile = {
            "attributes": {},
            "consensus_summary": f"{product_name} — extraction failed.",
            "reviewer_disagreements": "unknown",
            "sentiment": "unknown",
            "review_count": len(reviews),
        }

    summary = f"{product_name}: {profile.get('consensus_summary', '')}"
    cache.put_extraction(product_id, domain, profile, summary)
    return {"profile": profile, "summary": summary}
