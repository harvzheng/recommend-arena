"""LLM-as-Judge scoring logic — pointwise evaluation of candidates."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from .prompts import JUDGE_PROMPT, RUBRIC_HINTS

if TYPE_CHECKING:
    from shared.llm_provider import LLMProvider
    from .cache import JudgeCache

logger = logging.getLogger(__name__)


def query_hash(query_text: str) -> str:
    """Deterministic short hash of a query string."""
    return hashlib.sha256(query_text.encode()).hexdigest()[:16]


def judge_pointwise(
    user_query: str,
    domain: str,
    product_id: str,
    product_name: str,
    product_profile: dict,
    provider: "LLMProvider",
    cache: "JudgeCache",
) -> dict:
    """Score a single product against a user query (pointwise).

    Returns a dict with keys: score (1-10), reasoning, match_strengths,
    match_gaps.  Results are cached by (product_id, query_hash).
    """
    qh = query_hash(user_query)

    # Cache lookup
    cached = cache.get_judgment(product_id, qh)
    if cached is not None:
        return cached

    # Build profile JSON for the prompt (include rubric hints if available)
    profile_json = json.dumps(product_profile, indent=2)
    rubric = RUBRIC_HINTS.get(domain, "")
    if rubric:
        profile_json += f"\n\nRubric hints: {rubric}"

    prompt = JUDGE_PROMPT.format(
        domain=domain,
        user_query=user_query,
        product_name=product_name,
        product_profile_json=profile_json,
    )

    try:
        raw = provider.generate(prompt, json_mode=True)
        result = json.loads(raw)
    except (json.JSONDecodeError, RuntimeError) as exc:
        logger.warning(
            "Judge call failed for %s against query '%s': %s",
            product_id,
            user_query[:60],
            exc,
        )
        result = {
            "score": 1,
            "reasoning": f"Judge evaluation failed: {exc}",
            "match_strengths": [],
            "match_gaps": [],
        }

    # Clamp score to 1-10
    score = result.get("score", 1)
    if isinstance(score, (int, float)):
        result["score"] = max(1, min(10, int(score)))
    else:
        result["score"] = 1

    cache.put_judgment(product_id, qh, result)
    return result


def build_matched_attributes(judge_result: dict) -> dict[str, float]:
    """Convert judge match_strengths / match_gaps into a flat attribute dict.

    Strengths map directly via their ``confidence`` value (0-1).
    Gaps are inverted: severity 1.0 -> match 0.0.
    """
    attrs: dict[str, float] = {}
    for strength in judge_result.get("match_strengths", []):
        if isinstance(strength, dict) and "attribute" in strength:
            conf = strength.get("confidence", 0.5)
            if isinstance(conf, (int, float)):
                attrs[strength["attribute"]] = float(conf)
    for gap in judge_result.get("match_gaps", []):
        if isinstance(gap, dict) and "attribute" in gap:
            sev = gap.get("severity", 0.5)
            if isinstance(sev, (int, float)):
                attrs[gap["attribute"]] = 1.0 - float(sev)
    return attrs
