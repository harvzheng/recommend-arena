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
    """Picks providers per the configured tier.

    Slice 14.0 wires Tier 1 only — both providers are deterministic /
    local-cheap. Tier 2 and 3 land in the follow-up.
    """

    config: TierConfig

    def filter_parser(self) -> FilterParserProvider:
        if self.config.tier == "local":
            from implementations.design_14_local_hybrid.filter_parser import (
                parse_query as _parse_query,
            )

            class _LocalFilterParser:
                def parse(self, query_text: str, domain: str) -> list[dict]:
                    return _parse_query(query_text, domain)

            return _LocalFilterParser()
        # Tier 2 / 3 will route to a remote LLM with constrained JSON.
        # Until that lands, fall back to local so the runtime is usable.
        logger.warning(
            "tier=%s requested but only 'local' is wired in slice 14.0; "
            "using local filter parser.",
            self.config.tier,
        )
        return self.filter_parser_local_fallback()

    def filter_parser_local_fallback(self) -> FilterParserProvider:
        from implementations.design_14_local_hybrid.filter_parser import (
            parse_query as _parse_query,
        )

        class _Local:
            def parse(self, query_text: str, domain: str) -> list[dict]:
                return _parse_query(query_text, domain)

        return _Local()

    def explanation(self) -> ExplanationProvider:
        # Always deterministic in 14.0. Tier 2 / 3 land later.
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

        return _DeterministicExplainer()


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
