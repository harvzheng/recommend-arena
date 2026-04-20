"""Teacher labeling pipeline — prompt, scoring, caching."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.llm_provider import LLMProvider

from .context import ProductContext
from .db import get_cached_judgment, insert_teacher_judgment

logger = logging.getLogger(__name__)


@dataclass
class TeacherJudgment:
    """A single teacher evaluation of a (query, product) pair."""
    query: str
    product_id: str
    product_name: str
    score: float
    explanation: str
    matched_attributes: dict[str, float]
    teacher_model: str
    timestamp: str


TEACHER_PROMPT = """\
You are an expert {domain} product recommender. A user has described what they \
want, and you need to evaluate how well a specific product matches their preferences.

User's query: "{query}"

Product: {product_name}
{product_context}

Evaluate this product against the user's query. Consider:
- How well each mentioned attribute matches the user's stated preferences
- Trade-offs: where the product excels vs. falls short
- Overall suitability, accounting for how important each attribute is to the query

Respond with ONLY valid JSON in this exact format:
{{
  "score": <float 0.0 to 1.0, where 1.0 is a perfect match>,
  "explanation": "<2-4 sentences explaining why this score, citing specific product characteristics>",
  "matched_attributes": {{
    "<attribute_name>": <float 0.0 to 1.0 indicating match strength>,
    ...
  }}
}}

Scoring guidelines:
- 0.9-1.0: Near-perfect match on all queried attributes
- 0.7-0.89: Strong match with minor gaps
- 0.5-0.69: Decent match but notable trade-offs
- 0.3-0.49: Partial match, significant misalignments
- 0.0-0.29: Poor match, wrong category or contradicts preferences"""


def label_pair(
    query: str,
    product_ctx: ProductContext,
    llm: LLMProvider,
    domain: str,
) -> TeacherJudgment:
    prompt = TEACHER_PROMPT.format(
        domain=domain,
        query=query,
        product_name=product_ctx.product_name,
        product_context=product_ctx.context_text,
    )
    response = llm.generate(prompt, json_mode=True)

    # Small local models often produce malformed JSON (trailing commas,
    # markdown fences, etc). Use the same extraction logic as inference.py.
    from .inference import _extract_json
    try:
        json_str = _extract_json(response)
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Try fixing common issues: trailing commas before } or ]
        import re
        cleaned = re.sub(r",\s*([}\]])", r"\1", response)
        json_str = _extract_json(cleaned)
        parsed = json.loads(json_str)

    score = max(0.0, min(1.0, float(parsed["score"])))

    return TeacherJudgment(
        query=query,
        product_id=product_ctx.product_id,
        product_name=product_ctx.product_name,
        score=score,
        explanation=parsed["explanation"],
        matched_attributes={
            k: max(0.0, min(1.0, float(v)))
            for k, v in parsed.get("matched_attributes", {}).items()
        },
        teacher_model=llm.llm_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def label_all_pairs(
    queries: list[str],
    products: list[ProductContext],
    llm: LLMProvider,
    domain: str,
    db: sqlite3.Connection,
) -> list[TeacherJudgment]:
    judgments: list[TeacherJudgment] = []
    total = len(queries) * len(products)
    completed = 0

    for query in queries:
        for product in products:
            cached = get_cached_judgment(db, query, product.product_id)
            if cached:
                completed += 1
                continue

            try:
                judgment = label_pair(query, product, llm, domain)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    "Failed to label (%s, %s): %s — skipping",
                    query[:30], product.product_id, e,
                )
                completed += 1
                continue

            insert_teacher_judgment(
                db,
                query=query,
                product_id=judgment.product_id,
                score=judgment.score,
                explanation=judgment.explanation,
                matched_attributes=judgment.matched_attributes,
                teacher_model=judgment.teacher_model,
                created_at=judgment.timestamp,
            )
            db.commit()

            judgments.append(judgment)
            completed += 1

            if completed % 50 == 0:
                logger.info("Labeled %d/%d pairs", completed, total)

    logger.info("Labeling complete: %d new, %d total", len(judgments), total)
    return judgments
