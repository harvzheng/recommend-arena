"""Batch ABSA extraction and attribute normalization."""

from __future__ import annotations

import json
import logging

from shared.llm_provider import LLMProvider
from .prompts import ABSA_EXTRACTION_PROMPT
from .index import ProductRecord

logger = logging.getLogger(__name__)


def build_product_record(
    product: dict,
    reviews: list[dict],
    domain: str,
    llm: LLMProvider,
    attribute_names: list[str],
) -> ProductRecord:
    """Build a ProductRecord by extracting ABSA from reviews.

    Makes one LLM call per product with all its reviews batched together.
    Falls back to ground-truth attributes if extraction fails.
    """
    pid = product.get("product_id") or product.get("id", "")
    pname = product.get("product_name") or product.get("name", "")
    category = product.get("category", "")
    specs = product.get("metadata") or product.get("specs", {})
    gt_attributes = product.get("attributes", {})

    review_texts = [r.get("review_text") or r.get("text", "") for r in reviews]
    combined_reviews = "\n\n".join(
        f"Review {i+1}: {text}" for i, text in enumerate(review_texts) if text
    )

    # Attempt ABSA extraction via LLM
    review_attributes = {}
    if combined_reviews:
        review_attributes = _extract_absa(
            llm, pname, category, domain, attribute_names, specs, combined_reviews
        )

    # Build embedding text: product info + review highlights
    snippets = []
    for attr_data in review_attributes.values():
        if isinstance(attr_data, dict):
            snippets.extend(attr_data.get("snippets", []))
    snippet_text = " ".join(snippets[:10])

    embedding_text = (
        f"{pname}. Category: {category}. Domain: {domain}. "
        f"Attributes: {', '.join(f'{k}={v}' for k, v in gt_attributes.items() if isinstance(v, (int, float)))}. "
        f"Reviews: {snippet_text or combined_reviews[:500]}"
    )

    return ProductRecord(
        product_id=pid,
        product_name=pname,
        domain=domain,
        category=category,
        specs=specs,
        ground_truth_attributes=gt_attributes,
        review_attributes=review_attributes,
        review_text_combined=embedding_text,
    )


def _extract_absa(
    llm: LLMProvider,
    product_name: str,
    category: str,
    domain: str,
    attribute_names: list[str],
    specs: dict,
    reviews_text: str,
) -> dict:
    """Single LLM call to extract ABSA tuples for one product."""
    prompt = ABSA_EXTRACTION_PROMPT.format(
        product_name=product_name,
        category=category,
        domain=domain,
        attribute_names=", ".join(attribute_names),
        specs_json=json.dumps(specs),
        reviews_text=reviews_text.replace("{", "{{").replace("}", "}}"),
    )

    try:
        raw = llm.generate(prompt, json_mode=True)
        data = json.loads(raw)
        result = {}
        for attr_name, attr_data in data.get("attributes", {}).items():
            if isinstance(attr_data, dict) and attr_data.get("score") is not None:
                result[attr_name] = {
                    "score": float(attr_data["score"]),
                    "confidence": float(attr_data.get("confidence", 0.5)),
                    "snippets": attr_data.get("snippets", []),
                }
        # Merge additional attributes
        for attr_name, attr_data in data.get("additional_attributes", {}).items():
            if isinstance(attr_data, dict) and attr_data.get("score") is not None:
                result[attr_name] = {
                    "score": float(attr_data["score"]),
                    "confidence": float(attr_data.get("confidence", 0.3)),
                    "snippets": attr_data.get("snippets", []),
                }
        return result
    except Exception as e:
        logger.warning("ABSA extraction failed for %s: %s", product_name, e)
        return {}
