#!/usr/bin/env python3
"""discover_filter_phrases.py — Layer 2 of the filter automation.

Asks the teacher LLM (default: openrouter/owl-alpha) to generate
domain-specific phrase mappings from the catalog + (optionally) the eval
queries. Output is written to `artifacts/<domain>/filter_phrases.json`,
which the runtime filter parser loads alongside its hand-curated tables.

Usage:
    export RECOMMEND_LLM_PROVIDER=openai
    export OPENAI_API_KEY=$OPENROUTER_API_KEY
    export RECOMMEND_OPENAI_BASE_URL=https://openrouter.ai/api/v1
    export RECOMMEND_OPENAI_OPENAI_NO_JSON_FORMAT=1
    export RECOMMEND_LLM_MODEL=openrouter/owl-alpha
    python scripts/discover_filter_phrases.py --bundle artifacts/wine
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.llm_provider import get_provider  # noqa: E402

logger = logging.getLogger("discover_filter_phrases")


_PROMPT_TEMPLATE = """\
You design phrase->filter mappings for a domain-agnostic recommender.

The recommender's filter parser already handles:
  - explicit numeric ranges ("under $20", "over 95 points", "between $40 and $80")
  - direct category mentions (any literal value in the catalog: "malbec", "napa")
  - basic negation ("not sweet", "without oak")

Your job: generate the COLLOQUIAL/SUBJECTIVE phrase mappings the parser
DOESN'T cover. Examples of what to capture:
  - lifestyle phrases ("patio sipper", "weeknight wine")
  - food-pairing implications ("for steak", "with seafood")
  - taste/style descriptors ("bold", "elegant", "earthy", "fruity")
  - quality/reputation slang ("impress a sommelier", "crowd-pleaser")

For each phrase, emit one or more concrete filters using ONLY the
attributes and values shown in the schema below. A phrase can fan out
to multiple alternative filters (treated as OR by the prefilter).

== Domain: {domain} ==

== Schema ==
{schema_block}

== Eval queries (the kinds of phrases users actually use) ==
{eval_block}

== Output format ==
Return strict JSON, no prose, no markdown:
{{
  "phrases": [
    {{
      "phrase": "<lowercase phrase, exact substring users would type>",
      "filters": [
        {{"attribute": "<schema attr>", "op": "contains|gte|lte|eq", "value": <value>}}
      ]
    }}
  ]
}}

Rules:
  - Phrases must be 2-4 words, lowercase, EXACT substrings (no regex).
  - Skip anything already trivially extractable (single variety/country/price).
  - Aim for 30-60 entries: high-coverage on subjective vocabulary.
  - Each filter's value MUST exist in the catalog (or be a numeric in the
    observed range).
"""


def _summarize_schema(products: list[dict], max_values: int = 25) -> str:
    numeric: dict[str, list[float]] = defaultdict(list)
    categorical: dict[str, set[str]] = defaultdict(set)
    for p in products:
        for k, v in (p.get("attributes") or {}).items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                numeric[k].append(float(v))
            elif isinstance(v, str) and v.strip():
                categorical[k].add(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x.strip():
                        categorical[k].add(x)
    lines: list[str] = []
    for k, vals in sorted(numeric.items()):
        lo, hi = min(vals), max(vals)
        lines.append(f"  - {k} (numeric): range [{lo:.0f}, {hi:.0f}]")
    for k, vset in sorted(categorical.items()):
        if k == "taster":
            continue
        sample = sorted(vset)
        if len(sample) > max_values:
            sample = sample[:max_values] + [f"... ({len(vset) - max_values} more)"]
        lines.append(f"  - {k} (text): {', '.join(sample)}")
    return "\n".join(lines)


def _summarize_eval(eval_path: Path, n: int = 12) -> str:
    if not eval_path.exists():
        return "  (no eval set available)"
    data = json.loads(eval_path.read_text())
    queries = data.get("queries", [])
    sample = []
    for difficulty in ("vague", "hard", "medium", "easy"):
        for q in queries:
            if q.get("difficulty") == difficulty:
                sample.append(f"  - [{difficulty}] {q['query_text']}")
            if len(sample) >= n:
                break
        if len(sample) >= n:
            break
    return "\n".join(sample) if sample else "  (no queries found)"


def _validate_phrases(raw: list[dict], schema_attrs: dict[str, str],
                      categorical_values: dict[str, set[str]],
                      numeric_ranges: dict[str, tuple[float, float]]) -> list[dict]:
    """Drop entries that reference unknown attrs / values / out-of-range numerics."""
    clean: list[dict] = []
    seen_phrases: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        phrase = (entry.get("phrase") or "").strip().lower()
        filters = entry.get("filters") or []
        if not phrase or phrase in seen_phrases or len(phrase) < 3:
            continue
        if not isinstance(filters, list):
            continue
        valid: list[dict] = []
        for f in filters:
            if not isinstance(f, dict):
                continue
            attr = f.get("attribute")
            op = f.get("op")
            value = f.get("value")
            if attr not in schema_attrs:
                continue
            if op not in ("contains", "gte", "lte", "eq", "not_contains"):
                continue
            if schema_attrs[attr] == "text":
                if not isinstance(value, str) or not value.strip():
                    continue
                low_vals = {v.lower() for v in categorical_values.get(attr, set())}
                if value.lower() not in low_vals:
                    continue
            elif schema_attrs[attr] == "numeric":
                if not isinstance(value, (int, float)):
                    continue
                lo, hi = numeric_ranges.get(attr, (float("-inf"), float("inf")))
                if not (lo - 1 <= float(value) <= hi + 1):
                    continue
            valid.append({"attribute": attr, "op": op, "value": value})
        if valid:
            clean.append({"phrase": phrase, "filters": valid})
            seen_phrases.add(phrase)
    return clean


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="path to bundle dir")
    p.add_argument("--eval", default=None,
                   help="optional eval JSON to seed example queries")
    args = p.parse_args()

    bundle_path = Path(args.bundle).resolve()
    products_path = bundle_path / "products.jsonl"
    if not products_path.exists():
        logger.error("missing %s", products_path)
        return 2
    out_path = bundle_path / "filter_phrases.json"

    products = [json.loads(l) for l in products_path.read_text().splitlines() if l.strip()]
    schema_block = _summarize_schema(products)

    # Build attr schema for validation
    schema_attrs: dict[str, str] = {}
    categorical_values: dict[str, set[str]] = defaultdict(set)
    numeric_ranges: dict[str, tuple[float, float]] = {}
    numeric_buf: dict[str, list[float]] = defaultdict(list)
    for prod in products:
        for k, v in (prod.get("attributes") or {}).items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                schema_attrs[k] = "numeric"
                numeric_buf[k].append(float(v))
            elif isinstance(v, str) and v.strip():
                schema_attrs[k] = "text"
                categorical_values[k].add(v)
            elif isinstance(v, list):
                schema_attrs[k] = "text"
                for x in v:
                    if isinstance(x, str) and x.strip():
                        categorical_values[k].add(x)
    for k, vs in numeric_buf.items():
        numeric_ranges[k] = (min(vs), max(vs))

    domain = bundle_path.name
    eval_path = Path(args.eval).resolve() if args.eval else (
        _PROJECT_ROOT / "benchmark" / "data" / "per_domain" / f"{domain}_eval.json"
    )
    eval_block = _summarize_eval(eval_path)

    prompt = _PROMPT_TEMPLATE.format(domain=domain, schema_block=schema_block, eval_block=eval_block)
    logger.info("calling teacher LLM (model=%s)", get_provider().llm_model)
    raw_text = get_provider().generate(prompt, json_mode=True)
    text = _strip_json_fences(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("teacher returned non-JSON: %s\nraw=%s", e, raw_text[:400])
        return 1

    raw_phrases = data.get("phrases") or data.get("mappings") or []
    if not isinstance(raw_phrases, list):
        logger.error("expected a list of phrases, got: %r", type(raw_phrases))
        return 1

    clean = _validate_phrases(raw_phrases, schema_attrs, categorical_values, numeric_ranges)
    logger.info("kept %d / %d phrases after validation", len(clean), len(raw_phrases))
    out_path.write_text(json.dumps({"phrases": clean}, indent=2))
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
