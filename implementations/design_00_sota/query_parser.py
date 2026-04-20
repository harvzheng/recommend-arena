"""LLM-based query understanding and expansion."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from shared.llm_provider import LLMProvider
from .prompts import QUERY_PARSE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Structured representation of a natural language query."""
    desired_attributes: list[dict] = field(default_factory=list)
    # Each: {"name": str, "weight": float, "direction": "high"|"low"}
    negative_attributes: list[dict] = field(default_factory=list)
    # Each: {"name": str, "weight": float, "direction": "low"}
    spec_constraints: list[dict] = field(default_factory=list)
    # Each: {"field": str, "op": str, "value": float|str}
    categories: list[str] = field(default_factory=list)
    query_embedding_text: str = ""


def parse_query(
    llm: LLMProvider,
    query_text: str,
    domain: str,
    attribute_names: list[str],
    spec_fields: list[str],
    categories: list[str],
) -> ParsedQuery:
    """Parse a natural language query into structured search parameters."""
    prompt = QUERY_PARSE_PROMPT.format(
        domain=domain,
        attribute_names=", ".join(attribute_names),
        spec_fields=", ".join(spec_fields),
        categories=", ".join(categories),
        query_text=query_text.replace("{", "{{").replace("}", "}}"),
    )

    try:
        raw = llm.generate(prompt, json_mode=True)
        data = json.loads(raw)

        desired = []
        for attr in data.get("desired_attributes", []):
            if isinstance(attr, dict) and "name" in attr:
                desired.append({
                    "name": attr["name"],
                    "weight": float(attr.get("weight", 0.5)),
                    "direction": attr.get("direction", "high"),
                })

        negative = []
        for attr in data.get("negative_attributes", []):
            if isinstance(attr, dict) and "name" in attr:
                negative.append({
                    "name": attr["name"],
                    "weight": float(attr.get("weight", 0.5)),
                    "direction": attr.get("direction", "low"),
                })

        constraints = []
        for c in data.get("spec_constraints", []):
            if isinstance(c, dict) and "field" in c and "op" in c and "value" in c:
                constraints.append({
                    "field": c["field"],
                    "op": c["op"],
                    "value": c["value"],
                })

        cats = data.get("categories", [])
        if isinstance(cats, str):
            cats = [cats]

        embedding_text = data.get("query_embedding_text", query_text)

        return ParsedQuery(
            desired_attributes=desired,
            negative_attributes=negative,
            spec_constraints=constraints,
            categories=cats,
            query_embedding_text=embedding_text or query_text,
        )

    except Exception as e:
        logger.warning("Query parsing failed for '%s': %s", query_text, e)
        return ParsedQuery(query_embedding_text=query_text)
