"""ABSA extraction via LLM.

Uses the shared LLM provider to extract aspect-based sentiment triples
from review text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are an aspect-based sentiment analysis engine for {domain} product reviews.

Extract every product attribute mentioned in the review below. For each attribute, return:
- "aspect": a short noun phrase naming the attribute (e.g., "flex", "edge hold", "weight")
- "opinion": the reviewer's exact or paraphrased opinion about it
- "sentiment": a float from -1.0 (very negative) to 1.0 (very positive)

Rules:
- Only extract attributes about the product itself, not about shipping, price, or the reviewer.
- If the reviewer is comparative ("stiffer than X"), still extract the aspect with sentiment relative to positive.
- Return a JSON array. No commentary outside the JSON.

Review:
{review_text}
"""


def extract_aspects(
    review_text: str,
    domain: str,
    llm: LLMProvider,
) -> list[dict]:
    """Extract aspect-sentiment triples from a single review.

    Returns:
        List of dicts with keys: aspect, opinion, sentiment.
        Example: [{"aspect": "flex", "opinion": "very stiff", "sentiment": 0.85}]
    """
    prompt = EXTRACTION_PROMPT.format(domain=domain, review_text=review_text)
    try:
        raw = llm.generate(prompt, json_mode=True)
        aspects = _parse_response(raw)
        return aspects
    except Exception:
        logger.exception("ABSA extraction failed for review (len=%d)", len(review_text))
        return []


def _parse_response(raw: str) -> list[dict]:
    """Parse the LLM response into a list of aspect dicts.

    Handles both bare JSON arrays and JSON objects with a top-level key
    wrapping the array.
    """
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)

    # If the model returned {"aspects": [...]} or {"results": [...]}, unwrap
    if isinstance(data, dict):
        for key in ("aspects", "results", "attributes", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            # Single-item wrapped in a dict — treat as one-element list
            if "aspect" in data:
                data = [data]
            else:
                logger.warning("Unexpected JSON structure: %s", list(data.keys()))
                return []

    if not isinstance(data, list):
        return []

    validated: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        aspect = item.get("aspect", "").strip()
        if not aspect:
            continue
        sentiment = item.get("sentiment", 0.0)
        try:
            sentiment = float(sentiment)
        except (TypeError, ValueError):
            sentiment = 0.0
        sentiment = max(-1.0, min(1.0, sentiment))
        validated.append(
            {
                "aspect": aspect.lower(),
                "opinion": str(item.get("opinion", "")),
                "sentiment": sentiment,
            }
        )
    return validated
