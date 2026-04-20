"""Product context construction from reviews + specs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.llm_provider import LLMProvider


@dataclass
class ProductContext:
    """Pre-built text representation of a product for model input."""
    product_id: str
    product_name: str
    domain: str
    context_text: str
    spec_summary: str
    review_summary: str
    review_count: int
    metadata: dict = field(default_factory=dict)


_REVIEW_SUMMARY_PROMPT = (
    'Summarize what reviewers say about this {domain} product: '
    '"{product_name}".\n\n'
    'Reviews:\n{review_block}\n\n'
    'Write a 3-5 sentence summary covering the key attributes '
    'reviewers mention (performance, feel, strengths, weaknesses). '
    'Be specific -- use the reviewers\' language.'
)


def build_spec_summary(metadata: dict) -> str:
    if not metadata:
        return "No specs available."
    parts = []
    for key, value in metadata.items():
        label = key.replace("_", " ").title()
        parts.append(f"{label}: {value}")
    return ", ".join(parts)


def build_product_context(
    product: dict,
    reviews: list[dict],
    llm: LLMProvider,
    domain: str,
) -> ProductContext:
    pid = product.get("product_id", product.get("id", ""))
    pname = product.get("product_name", product.get("name", ""))
    metadata = product.get("metadata", product.get("specs", {}))

    spec_summary = build_spec_summary(metadata)

    review_texts = [
        r.get("review_text", r.get("text", "")) for r in reviews
    ]
    review_texts = [t for t in review_texts if t]

    if review_texts:
        review_block = "\n---\n".join(review_texts[:20])
        prompt = _REVIEW_SUMMARY_PROMPT.format(
            domain=domain,
            product_name=pname,
            review_block=review_block,
        )
        review_summary = llm.generate(prompt)
    else:
        review_summary = "No reviews available."

    context_text = (
        f"Specs: {spec_summary}\n\n"
        f"Review consensus ({len(review_texts)} reviews): {review_summary}"
    )

    return ProductContext(
        product_id=pid,
        product_name=pname,
        domain=domain,
        context_text=context_text,
        spec_summary=spec_summary,
        review_summary=review_summary,
        review_count=len(review_texts),
        metadata=metadata,
    )
