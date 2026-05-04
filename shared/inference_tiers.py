"""Tier router for design 14's runtime.

Three tiers (per the design 14 spec):

  Tier 1 — fully local. Filter parser + explanation run locally
           (deterministic regex / Qwen3-4B). Retrieval + rerank local.
           $0 marginal cost. Latency ~700ms-1s.

  Tier 2 — hybrid. Retrieval + rerank stay local. Filter parsing and
           explanation route to Together AI's Llama-3.1-8B
           (~$0.18/1M tokens). Latency ~1-2s including network.

  Tier 3 — frontier escalation. Same as tier 2, but route to Opus 4.7
           when the reranker's top-1 vs top-2 score margin is below
           threshold (the genuinely hard queries). ~5% of traffic.

Slice 14.0 ships Tier 1 only. This module exists so the runtime has a
single seam to grow tiers 2 and 3 against, and so the recommender can
expose a `tier=` config knob from day one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

Tier = Literal["local", "hybrid", "frontier"]
DEFAULT_TIER: Tier = "local"

# When the reranker's top-1 vs top-2 score gap (as a fraction of the
# top-1 score) is below this threshold, tier 3 escalates to Opus 4.7.
# Tuned empirically on the existing arena's 'hard' bucket; revisit when
# we have more eval data.
ESCALATION_MARGIN_THRESHOLD = 0.05


@dataclass
class TierConfig:
    """Runtime configuration for the tier router."""

    tier: Tier = DEFAULT_TIER
    # Tier 2 only: the Together / Anthropic-compatible base URL. Read
    # from env if not passed; the runtime can stay fully local even
    # when this is unset because Tier 1 ignores it.
    together_api_key: str | None = None
    anthropic_api_key: str | None = None
    # Tier 3 only: how aggressive the escalation is. Smaller threshold
    # → fewer escalations → cheaper but worse on hard queries.
    escalation_margin: float = ESCALATION_MARGIN_THRESHOLD

    @classmethod
    def from_env(cls) -> "TierConfig":
        return cls(
            tier=_parse_tier(os.environ.get("ARENA_TIER", DEFAULT_TIER)),
            together_api_key=os.environ.get("TOGETHER_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            escalation_margin=float(
                os.environ.get("ARENA_ESCALATION_MARGIN", ESCALATION_MARGIN_THRESHOLD)
            ),
        )


def _parse_tier(s: str) -> Tier:
    s = s.strip().lower()
    if s in ("local", "hybrid", "frontier"):
        return s  # type: ignore[return-value]
    logger.warning("ARENA_TIER=%r is not recognized; defaulting to %r", s, DEFAULT_TIER)
    return DEFAULT_TIER


# ---------------------------------------------------------------------------
# Provider protocols. Concrete implementations land in a follow-up — the
# point of this module right now is to nail down the seams.
# ---------------------------------------------------------------------------
class FilterParserProvider(Protocol):
    """Extracts structured filter constraints from natural-language queries."""

    def parse(self, query_text: str, domain: str) -> list[dict]:
        ...


class ExplanationProvider(Protocol):
    """Generates the human-readable why-this-product explanation."""

    def explain(
        self,
        query_text: str,
        product: dict,
        matched_attributes: dict[str, float],
    ) -> str:
        ...


@dataclass
class TierRouter:
    """Picks providers per the configured tier."""

    config: TierConfig

    def filter_parser(self) -> FilterParserProvider:
        if self.config.tier == "local":
            return _DeterministicFilterParser()
        # Tier 2 / 3 use the existing shared.llm_provider abstraction.
        # That's the same surface the existing LLM-using designs sit on,
        # so the routing logic is concentrated rather than duplicated
        # against three different SDKs.
        try:
            return _LLMFilterParser()
        except Exception as e:
            logger.warning(
                "LLM filter parser unavailable (%s); falling back to local.", e
            )
            return _DeterministicFilterParser()

    def filter_parser_local_fallback(self) -> FilterParserProvider:
        return _DeterministicFilterParser()

    def explanation(self) -> ExplanationProvider:
        if self.config.tier == "local":
            return _DeterministicExplainer()
        try:
            return _LLMExplainer()
        except Exception as e:
            logger.warning(
                "LLM explainer unavailable (%s); falling back to local.", e
            )
            return _DeterministicExplainer()


# ---------------------------------------------------------------------------
# Tier 1 — deterministic, no model load.
# ---------------------------------------------------------------------------
class _DeterministicFilterParser:
    """Wraps the existing regex/keyword filter parser as a Provider."""

    def parse(self, query_text: str, domain: str) -> list[dict]:
        # Lazy import to avoid a circular dep at module-load time.
        from implementations.design_14_local_hybrid.filter_parser import (
            parse_query,
        )
        return parse_query(query_text, domain)


class _DeterministicExplainer:
    def explain(
        self,
        query_text: str,
        product: dict,
        matched_attributes: dict[str, float],
    ) -> str:
        if not matched_attributes:
            return f"Top match for {query_text!r}."
        cols = ", ".join(sorted(matched_attributes.keys())[:3])
        return f"Matched on {cols}; combined retrieval + rerank."


# ---------------------------------------------------------------------------
# Tier 2 — LLM-based, via shared.llm_provider. Tier 3 reuses these and
# only differs in the escalation router (see should_escalate).
# ---------------------------------------------------------------------------
class _LLMFilterParser:
    """LLM-backed filter parser. Calls the configured LLMProvider with a
    JSON-mode prompt that asks for the same {attribute, op, value} shape
    the deterministic parser emits.

    The LLM request is best-effort; we fall back to the deterministic
    parser if the response doesn't match the expected shape.
    """

    SYSTEM = (
        "You extract structured filter constraints from a product search "
        "query. Output a JSON array of objects shaped like "
        "{\"attribute\": \"<field>\", \"op\": \"<eq|gte|lte|contains|not_contains|in>\", "
        "\"value\": <number-or-string-or-list>}. "
        "Only emit attributes you can ground in the catalog. Return [] "
        "if the query has no extractable constraints. JSON only, no prose."
    )

    def __init__(self) -> None:
        from shared.llm_provider import get_provider
        self._provider = get_provider()

    def parse(self, query_text: str, domain: str) -> list[dict]:
        import json
        prompt = (
            f"{self.SYSTEM}\n\n"
            f"Domain: {domain}\nQuery: {query_text}\n\n"
            f"JSON array:"
        )
        try:
            text = self._provider.generate(prompt, json_mode=True)
        except Exception as e:
            logger.warning("LLM filter parser call failed (%s); local fallback", e)
            return _DeterministicFilterParser().parse(query_text, domain)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _DeterministicFilterParser().parse(query_text, domain)
        if not isinstance(data, list):
            return _DeterministicFilterParser().parse(query_text, domain)

        out: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("attribute", "op", "value")):
                continue
            out.append({
                "attribute": str(item["attribute"]),
                "op": str(item["op"]),
                "value": item["value"],
            })
        return out


class _LLMExplainer:
    """LLM-backed explainer. Generates a single-sentence rationale."""

    SYSTEM = (
        "You write one short sentence explaining why a product matches "
        "a user's query. Reference concrete attributes that appear in "
        "the product data — never invent attributes the product doesn't "
        "have. Plain text only."
    )

    def __init__(self) -> None:
        from shared.llm_provider import get_provider
        self._provider = get_provider()

    def explain(
        self,
        query_text: str,
        product: dict,
        matched_attributes: dict[str, float],
    ) -> str:
        import json
        prompt = (
            f"{self.SYSTEM}\n\n"
            f"Query: {query_text}\n"
            f"Product:\n{json.dumps(product, indent=2)[:4000]}\n"
            f"Matched attributes: {sorted(matched_attributes.keys())}\n\n"
            f"Sentence:"
        )
        try:
            return self._provider.generate(prompt, json_mode=False).strip()
        except Exception as e:
            logger.warning("LLM explainer call failed (%s); local fallback", e)
            return _DeterministicExplainer().explain(
                query_text, product, matched_attributes
            )


def should_escalate(reranker_scores: list[float], cfg: TierConfig) -> bool:
    """Return True iff the top-1/top-2 margin is below the threshold.

    Empty / single-result rankings are treated as "high confidence"
    and don't escalate — there's nothing to compare against.
    """
    if cfg.tier != "frontier":
        return False
    if len(reranker_scores) < 2:
        return False
    top1, top2 = reranker_scores[0], reranker_scores[1]
    if top1 <= 0:
        return False
    return (top1 - top2) / top1 < cfg.escalation_margin
