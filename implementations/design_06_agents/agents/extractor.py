"""Extractor Agent -- parses reviews into structured attributes per product.

Runs during ingestion. Groups reviews by product and uses LLM to extract
sentiment per attribute. Falls back to simple heuristics if LLM fails.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from ..state import ProductAttributes

logger = logging.getLogger(__name__)

# Attributes we look for per domain
DOMAIN_ATTRIBUTES = {
    "ski": [
        "stiffness", "damp", "edge_grip", "stability_at_speed",
        "playfulness", "powder_float", "forgiveness",
    ],
    "running_shoe": [
        "cushioning", "responsiveness", "stability", "grip",
        "breathability", "durability", "weight_feel",
    ],
}

EXTRACT_PROMPT = """You are analyzing {review_count} reviews for a {domain} product: "{product_name}".

Reviews:
{numbered_reviews}

For each of the following attributes, rate the overall reviewer sentiment from -1.0 (very negative) to 1.0 (very positive). Only include attributes that reviewers actually discuss. If an attribute is not mentioned, omit it.

Attributes to evaluate: {attributes}

Return ONLY valid JSON in this exact format (no markdown, no commentary):
{{"attribute_name": score, ...}}

Example: {{"stiffness": 0.8, "edge_grip": 0.9, "forgiveness": -0.5}}"""


def extract_agent(
    product_data: dict,
    reviews: list[dict],
    domain: str,
    llm_provider=None,
) -> ProductAttributes:
    """Extract structured attributes from a product and its reviews.

    Args:
        product_data: Raw product dict from benchmark data.
        reviews: List of review dicts for this product.
        domain: Product domain (e.g. "ski").
        llm_provider: Shared LLM provider instance.

    Returns:
        ProductAttributes with extracted sentiment scores.
    """
    # Build base product from structured data
    product = ProductAttributes(
        product_id=product_data["id"],
        product_name=product_data["name"],
        domain=domain,
        brand=product_data.get("brand", ""),
        category=product_data.get("category", ""),
        specs=product_data.get("specs", {}),
        attributes=product_data.get("attributes", {}),
        review_count=len(reviews),
        avg_rating=(
            sum(r.get("rating", 0) for r in reviews) / len(reviews)
            if reviews
            else 0.0
        ),
    )

    # Extract sentiment from reviews
    if reviews and llm_provider is not None:
        sentiment = _extract_sentiment_llm(
            product_data, reviews, domain, llm_provider
        )
        if sentiment:
            product.sentiment_scores = sentiment
        else:
            product.sentiment_scores = _extract_sentiment_heuristic(
                reviews, domain
            )
    elif reviews:
        product.sentiment_scores = _extract_sentiment_heuristic(reviews, domain)

    return product


def _extract_sentiment_llm(
    product_data: dict,
    reviews: list[dict],
    domain: str,
    llm_provider,
) -> dict[str, float]:
    """Use LLM to extract per-attribute sentiment from reviews."""
    attributes = DOMAIN_ATTRIBUTES.get(domain, [])
    if not attributes:
        return {}

    numbered = "\n".join(
        f"{i+1}. [{r.get('reviewer', 'anon')}] (rating: {r.get('rating', '?')}/5): "
        f"{r.get('text', '')}"
        for i, r in enumerate(reviews)
    )

    prompt = EXTRACT_PROMPT.format(
        review_count=len(reviews),
        domain=domain,
        product_name=product_data["name"],
        numbered_reviews=numbered,
        attributes=", ".join(attributes),
    )

    max_retries = 2
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = llm_provider.generate(prompt, json_mode=True)
            # Clean response -- strip markdown fences if present
            cleaned = _clean_json_response(response)
            parsed = json.loads(cleaned)

            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict, got {type(parsed)}")

            # Validate and clamp values
            sentiment = {}
            for attr, score in parsed.items():
                if attr in attributes and isinstance(score, (int, float)):
                    sentiment[attr] = max(-1.0, min(1.0, float(score)))

            return sentiment

        except (json.JSONDecodeError, ValueError, KeyError, RuntimeError) as e:
            last_error = str(e)
            logger.warning(
                "Sentiment extraction attempt %d failed for %s: %s",
                attempt + 1,
                product_data["name"],
                last_error,
            )
            if isinstance(e, RuntimeError):
                # Provider is down, no point retrying
                break

    logger.warning(
        "All sentiment extraction attempts failed for %s, using heuristic",
        product_data["name"],
    )
    return {}


def _extract_sentiment_heuristic(
    reviews: list[dict], domain: str
) -> dict[str, float]:
    """Simple keyword-based sentiment extraction as fallback."""
    attributes = DOMAIN_ATTRIBUTES.get(domain, [])
    if not attributes:
        return {}

    # Simple positive/negative keyword mapping
    positive_words = {
        "excellent", "incredible", "amazing", "outstanding", "perfect",
        "love", "great", "best", "fantastic", "phenomenal", "superb",
        "impressive", "solid", "strong", "good", "nice",
    }
    negative_words = {
        "terrible", "awful", "poor", "bad", "weak", "horrible",
        "disappointing", "mediocre", "struggles", "lacks", "minimal",
        "no", "not", "none", "zero", "worst",
    }

    # Attribute synonyms for keyword matching
    attr_keywords: dict[str, set[str]] = {
        "stiffness": {"stiff", "stiffness", "flex", "rigid"},
        "damp": {"damp", "dampness", "smooth", "vibration", "chatter"},
        "edge_grip": {"edge", "grip", "hold", "carve", "ice", "hardpack"},
        "stability_at_speed": {"stable", "stability", "speed", "fast"},
        "playfulness": {"playful", "fun", "lively", "poppy", "butter"},
        "powder_float": {"powder", "float", "deep", "soft snow"},
        "forgiveness": {"forgiving", "forgiveness", "easy", "friendly"},
        "cushioning": {"cushion", "cushioning", "soft", "plush", "padding"},
        "responsiveness": {"responsive", "snappy", "bouncy", "energy return"},
        "stability": {"stable", "stability", "support"},
        "grip": {"grip", "traction", "outsole"},
        "breathability": {"breathable", "breathability", "ventilation", "hot"},
        "durability": {"durable", "durability", "wear", "last"},
        "weight_feel": {"light", "lightweight", "heavy", "weight"},
    }

    sentiment: dict[str, list[float]] = defaultdict(list)

    for review in reviews:
        text = review.get("text", "").lower()
        words = set(re.findall(r"\w+", text))

        for attr in attributes:
            keywords = attr_keywords.get(attr, {attr})
            if not keywords & words:
                continue

            # Simple context-free sentiment
            pos_count = len(positive_words & words)
            neg_count = len(negative_words & words)
            total = pos_count + neg_count
            if total > 0:
                score = (pos_count - neg_count) / total
            else:
                # Neutral mention
                score = 0.0

            # Boost by rating
            rating = review.get("rating", 3)
            rating_factor = (rating - 3) / 2  # -1 to 1
            combined = 0.6 * score + 0.4 * rating_factor

            sentiment[attr].append(combined)

    return {
        attr: sum(scores) / len(scores)
        for attr, scores in sentiment.items()
        if scores
    }


def _clean_json_response(text: str) -> str:
    """Strip markdown code fences and other wrappers from LLM JSON output."""
    text = text.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
