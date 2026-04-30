"""Phase 1: synthetic query generation for the teacher to label.

See design-13 spec §4. The benchmark queries (queries.json) MUST be
excluded to keep the held-out evaluation set held out (spec §4.3).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

PROMPT_TEMPLATE = """\
You are generating product search queries that real users would type.
Domain: {domain}
Difficulty bucket: {bucket}
Attribute focus: {attributes}

Easy queries: single clear attribute, plain language.
Medium queries: 2-3 constraints, may mix attributes and metadata.
Hard queries: include negations, ranges, or trade-offs.
Vague queries: subjective or metaphorical, no explicit attribute names.
Cross-domain queries: ambiguous about which domain they target.

Generate {n} distinct queries. Avoid templated phrasing.
Vary length, tone, and word choice.

Return JSON only: {{"queries": [{{"text": "...", "rationale": "..."}}, ...]}}
"""

# (queries_per_combo, combo_size). A combo_size of 0 means "no attribute focus".
_BUCKET_PLAN: dict[str, tuple[int, int]] = {
    "easy":         (5, 1),
    "medium":       (5, 2),
    "hard":         (5, 2),
    "vague":        (5, 0),
    "cross_domain": (3, 0),
}

_DEFAULT_COMBO_CAPS: dict[str, int] = {
    "easy": 8, "medium": 16, "hard": 12, "vague": 10, "cross_domain": 5,
}


@dataclass
class SyntheticQuery:
    query_id: str
    text: str
    difficulty: str
    seed_attributes: list[str] = field(default_factory=list)
    domain: str = ""


def _normalize_query(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!?,;:")


def _build_combos(
    attributes: list[str],
    combo_size: int,
    cap: int,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    if combo_size == 0:
        return [()] * cap
    all_combos = list(combinations(attributes, combo_size))
    rng.shuffle(all_combos)
    return all_combos[:cap]


def generate_synthetic_queries(
    domain: str,
    attributes: list[str],
    llm,
    seed: int = 42,
    benchmark_queries: Iterable[str] | None = None,
    buckets: tuple[str, ...] = ("easy", "medium", "hard", "vague", "cross_domain"),
    max_combos_per_bucket: int | None = None,
) -> list[SyntheticQuery]:
    rng = random.Random(seed)
    benchmark_norm = {_normalize_query(q) for q in (benchmark_queries or set())}
    out: list[SyntheticQuery] = []
    counter = 0
    for bucket in buckets:
        n_per_combo, combo_size = _BUCKET_PLAN[bucket]
        cap = max_combos_per_bucket if max_combos_per_bucket is not None else _DEFAULT_COMBO_CAPS[bucket]
        combos = _build_combos(attributes, combo_size, cap, rng)
        for combo in combos:
            prompt = PROMPT_TEMPLATE.format(
                domain=domain, bucket=bucket,
                attributes=list(combo), n=n_per_combo,
            )
            raw = llm.generate(prompt)
            parsed = json.loads(raw)
            for entry in parsed.get("queries", []):
                text = entry["text"].strip()
                if _normalize_query(text) in benchmark_norm:
                    continue
                counter += 1
                out.append(SyntheticQuery(
                    query_id=f"syn-{counter:04d}",
                    text=text,
                    difficulty=bucket,
                    seed_attributes=list(combo),
                    domain=domain,
                ))
    return out
