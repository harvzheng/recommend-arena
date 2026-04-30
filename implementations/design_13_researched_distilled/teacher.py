"""Phase 2: researched teacher labeling.

For each (synthetic_query, product), call the teacher (ResearchedLLM with
web_search) once. Cache results in SQLite so the long, expensive run is
resumable. See design-13 spec §5.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from implementations.design_13_researched_distilled.synthetic_queries import (
    SyntheticQuery,
)

logger = logging.getLogger(__name__)

TEACHER_PROMPT = """\
You are an expert {domain} product evaluator. Judge how well a product
matches a user's query. You may use web_search to look up authoritative
reviews; cap research at 2 search calls + 1 fetch (tool budget enforced
by the runtime). Resolve any jargon in the query.

Query: {query_text}

Product: {product_name}
Local context (specs + reviews from our catalog):
{product_context}

Score rubric:
0.9-1.0: near-perfect on every queried attribute
0.7-0.89: strong with minor gaps
0.5-0.69: decent with notable trade-offs
0.3-0.49: partial, significant misalignment
0.0-0.29: poor or wrong category

Output JSON only:
{{
  "score": float,
  "matched_attributes": {{"<attr>": float, ...}},
  "explanation": "<2-4 sentences>",
  "evidence": [{{"source": str, "text": str, "relevance": float}}, ...]
}}
"""


@dataclass
class TeacherJudgment:
    query_id: str
    product_id: str
    score: float
    matched_attributes: dict[str, float]
    explanation: str
    evidence: list[dict] = field(default_factory=list)
    teacher_model: str = ""
    research_calls: int = 0
    created_at: str = ""


def _extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unterminated JSON object")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def label_pair(
    query: SyntheticQuery,
    product_name: str,
    product_id: str,
    product_context: str,
    domain: str,
    llm,
    max_tool_calls: int = 3,
) -> TeacherJudgment:
    prompt = TEACHER_PROMPT.format(
        domain=domain,
        query_text=query.text,
        product_name=product_name,
        product_context=product_context,
    )
    resp = llm.research_generate(prompt=prompt, max_tool_calls=max_tool_calls)
    try:
        parsed = json.loads(_extract_json(resp.text))
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"teacher returned unparseable response: {e}") from e

    return TeacherJudgment(
        query_id=query.query_id,
        product_id=product_id,
        score=_clamp(parsed["score"]),
        matched_attributes={
            k: _clamp(v) for k, v in parsed.get("matched_attributes", {}).items()
        },
        explanation=str(parsed.get("explanation", "")),
        evidence=list(parsed.get("evidence", [])),
        teacher_model=getattr(llm, "model", "unknown"),
        research_calls=getattr(resp, "tool_calls", 0),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def label_all_pairs(
    queries: list[SyntheticQuery],
    products: list[dict],
    domain: str,
    llm,
    conn: sqlite3.Connection,
    max_tool_calls: int = 3,
) -> list[TeacherJudgment]:
    out: list[TeacherJudgment] = []
    teacher_model = getattr(llm, "model", "unknown")
    total = len(queries) * len(products)
    done = 0
    for q in queries:
        for p in products:
            cached = conn.execute(
                "SELECT 1 FROM teacher_judgments "
                "WHERE query_id=? AND product_id=? AND teacher_model=?",
                (q.query_id, p["product_id"], teacher_model),
            ).fetchone()
            if cached:
                done += 1
                continue
            j = label_pair(
                query=q,
                product_name=p["product_name"],
                product_id=p["product_id"],
                product_context=p["context_text"],
                domain=domain,
                llm=llm,
                max_tool_calls=max_tool_calls,
            )
            conn.execute(
                "INSERT INTO teacher_judgments(query_id, product_id, score, "
                "matched_attributes_json, explanation, evidence_json, "
                "teacher_model, research_calls, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    j.query_id, j.product_id, j.score,
                    json.dumps(j.matched_attributes),
                    j.explanation,
                    json.dumps(j.evidence),
                    j.teacher_model,
                    j.research_calls,
                    j.created_at,
                ),
            )
            conn.commit()
            out.append(j)
            done += 1
            if done % 50 == 0:
                logger.info("teacher: labeled %d/%d", done, total)
    return out
