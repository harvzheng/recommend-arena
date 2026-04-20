"""Prompt templates for the LLM-as-Judge recommendation system."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Extraction prompt — used during ingestion to build structured profiles
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """\
You are analyzing {review_count} user reviews for a {domain} product: "{product_name}".

Extract a structured profile capturing what reviewers collectively say about this product.
Identify the key attributes that matter for {domain} products.
Note where reviewers agree and disagree.

Reviews:
{reviews_text}

Respond in JSON matching this structure:
{{
  "attributes": {{...key-value pairs of product characteristics...}},
  "consensus_summary": "2-3 sentence summary of what reviewers agree on",
  "reviewer_disagreements": "areas where reviewers disagree, or 'none'",
  "sentiment": "positive|mixed|negative",
  "review_count": {review_count}
}}

Respond ONLY with valid JSON. No markdown fences, no commentary."""

# ---------------------------------------------------------------------------
# Judge prompt — pointwise scoring of a single candidate
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """\
You are evaluating whether a {domain} product matches a user's preferences.

User's query: "{user_query}"

Product: {product_name}
Profile:
{product_profile_json}

Score this product's match to the user's query on a scale of 1-10.
Explain your reasoning in 2-3 sentences.
For each match_strength, also provide a numeric confidence from 0.0 to 1.0.

Respond in JSON:
{{"score": <1-10>, "reasoning": "...", "match_strengths": [{{"attribute": "...", "confidence": 0.0-1.0}}, ...], "match_gaps": [{{"attribute": "...", "severity": 0.0-1.0}}, ...]}}

Respond ONLY with valid JSON. No markdown fences, no commentary."""

# ---------------------------------------------------------------------------
# Domain-specific rubric hints (soft guidance for the judge)
# ---------------------------------------------------------------------------
RUBRIC_HINTS: dict[str, str] = {
    "ski": (
        "Consider: flex/stiffness match, terrain suitability, width "
        "appropriateness, skill level alignment, weight, dampness, "
        "edge grip, playfulness vs stability"
    ),
    "running_shoe": (
        "Consider: cushion level match, responsiveness, plate presence, "
        "drop height, intended pace, weight, durability"
    ),
    "cookie": (
        "Consider: texture match, flavor profile, dietary restrictions, "
        "freshness indicators"
    ),
}
