"""LLM-generated synthetic training data for the LTR ranker.

Uses the shared LLM provider to generate (query, product, relevance) triples
that serve as training data for the XGBoost ranker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic query templates per domain
# ---------------------------------------------------------------------------

DOMAIN_QUERIES: dict[str, list[str]] = {
    "ski": [
        "stiff carving ski for advanced skiers",
        "forgiving beginner ski for groomed runs",
        "all-mountain ski that handles everything",
        "lightweight powder ski with good float",
        "stable high-speed frontside ski",
        "playful freestyle ski for park and trees",
        "damp ski for icy East Coast conditions",
        "wide freeride ski for off-piste",
        "race ski with maximum edge grip",
        "versatile one-ski quiver",
        "stiff GS race ski 180cm+",
        "soft forgiving ski for learning",
        "powder ski with rocker for deep snow",
        "carving ski with good dampness",
        "touring ski that is lightweight",
        "freeride charger for variable conditions",
        "playful all-mountain ski not too stiff",
        "precise on-piste ski with strong edge hold",
        "beginner friendly ski with easy turn initiation",
        "expert ski for steep terrain and big mountains",
    ],
    "running_shoe": [
        "cushioned daily trainer for long runs",
        "lightweight racing flat for 5K",
        "trail running shoe with good grip",
        "stability shoe for overpronation",
        "comfortable recovery shoe",
        "responsive tempo shoe",
        "durable shoe for high mileage",
        "waterproof trail shoe",
        "minimalist shoe with low drop",
        "carbon plated race shoe",
    ],
}

# Fallback queries for any domain
DEFAULT_QUERIES = [
    "best overall product",
    "budget-friendly option",
    "premium high-performance choice",
    "most versatile option",
    "beginner-friendly product",
    "advanced expert-level product",
    "lightweight and responsive",
    "durable and long-lasting",
    "most comfortable option",
    "best value for the price",
]

JUDGMENT_PROMPT = """You are evaluating product relevance for a search query.

Query: {query}
Product: {product_name}
Brand: {brand}
Category: {category}
Top attributes from reviews (aspect: avg_sentiment):
{attributes}

Rate relevance on a scale of 0-3:
  0 = completely irrelevant
  1 = marginally relevant (partially matches but missing key aspects)
  2 = relevant, good match (matches most criteria)
  3 = excellent match, exactly what was asked for

Respond with JSON only: {{"score": <int 0-3>, "reason": "<one sentence>"}}"""


def _get_training_data_path(domain: str) -> Path:
    """Return the path for persisted training data."""
    base = Path(__file__).parent / "training_data"
    return base / f"synthetic_judgments_{domain}.json"


def _compute_corpus_hash(products: list[dict]) -> str:
    """Compute a hash of the product corpus for staleness detection."""
    ids = sorted(p.get("product_id", p.get("id", "")) for p in products)
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]


def _stratified_sample(products: list[dict], n: int) -> list[dict]:
    """Sample n products with a mix of categories for diversity."""
    if len(products) <= n:
        return list(products)

    by_cat: dict[str, list[dict]] = {}
    for p in products:
        cat = p.get("category", "other")
        by_cat.setdefault(cat, []).append(p)

    sampled = []
    cats = list(by_cat.values())
    random.shuffle(cats)
    idx = 0
    while len(sampled) < n:
        cat = cats[idx % len(cats)]
        if cat:
            sampled.append(cat.pop(random.randrange(len(cat))))
        idx += 1
        if all(len(c) == 0 for c in cats):
            break
    return sampled


def load_or_generate_judgments(
    products: list[dict],
    extracted_attributes: dict[str, list[dict]],
    domain: str,
    llm: LLMProvider,
    products_per_query: int = 15,
    force_regenerate: bool = False,
) -> list[dict]:
    """Load cached training data or generate new synthetic judgments.

    Returns:
        List of {query, product_id, relevance, reason} dicts.
    """
    data_path = _get_training_data_path(domain)
    corpus_hash = _compute_corpus_hash(products)

    # Try loading cached data
    if not force_regenerate and data_path.exists():
        try:
            cached = json.loads(data_path.read_text())
            if cached.get("corpus_hash") == corpus_hash:
                logger.info(
                    "Loaded %d cached judgments for domain '%s'",
                    len(cached["judgments"]),
                    domain,
                )
                return cached["judgments"]
            logger.info("Corpus hash mismatch, regenerating training data")
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to load cached data, regenerating")

    # Generate new judgments
    queries = DOMAIN_QUERIES.get(domain, DEFAULT_QUERIES)
    judgments = _generate_judgments(
        queries, products, extracted_attributes, llm, products_per_query
    )

    # Persist
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(
            {
                "corpus_hash": corpus_hash,
                "domain": domain,
                "query_count": len(queries),
                "products_per_query": products_per_query,
                "total_judgments": len(judgments),
                "judgments": judgments,
            },
            indent=2,
        )
    )
    logger.info(
        "Generated and saved %d judgments for domain '%s' to %s",
        len(judgments),
        domain,
        data_path,
    )
    return judgments


def _generate_judgments(
    queries: list[str],
    products: list[dict],
    extracted_attributes: dict[str, list[dict]],
    llm: LLMProvider,
    products_per_query: int,
) -> list[dict]:
    """Generate (query, product, relevance) triples via LLM."""
    judgments = []

    for qi, query in enumerate(queries):
        sampled = _stratified_sample(products, products_per_query)

        for product in sampled:
            pid = product.get("product_id", product.get("id", ""))
            attrs = extracted_attributes.get(pid, [])
            attr_str = "\n".join(
                f"  {a['aspect']}: {a.get('avg_sentiment', a.get('sentiment', 0)):.2f}"
                for a in attrs[:5]
            )

            prompt = JUDGMENT_PROMPT.format(
                query=query,
                product_name=product.get("product_name", product.get("name", "")),
                brand=product.get("brand", "unknown"),
                category=product.get("category", "unknown"),
                attributes=attr_str or "  (no attributes extracted)",
            )

            try:
                response = llm.generate(prompt, json_mode=True)
                parsed = json.loads(response)
                score = int(parsed.get("score", 0))
                score = max(0, min(3, score))  # clamp
                reason = parsed.get("reason", "")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    "Failed to parse LLM judgment for query=%r, product=%s: %s",
                    query,
                    pid,
                    e,
                )
                # Fallback: use a heuristic score
                score = _heuristic_relevance(query, product, attrs)
                reason = "heuristic fallback"

            judgments.append(
                {
                    "query": query,
                    "product_id": pid,
                    "relevance": score,
                    "reason": reason,
                }
            )

        logger.debug(
            "Generated judgments for query %d/%d: %r",
            qi + 1,
            len(queries),
            query,
        )

    return judgments


def _heuristic_relevance(
    query: str, product: dict, attrs: list[dict]
) -> int:
    """Compute a simple heuristic relevance score as fallback.

    Uses keyword overlap between query and product name/category/attributes.
    """
    query_lower = query.lower()
    name = product.get("product_name", product.get("name", "")).lower()
    category = product.get("category", "").lower()

    score = 0
    query_words = set(query_lower.split())

    # Name overlap
    name_words = set(name.split())
    overlap = len(query_words & name_words)
    if overlap > 0:
        score += 1

    # Category match
    for word in query_words:
        if word in category:
            score += 1
            break

    # Attribute match
    attr_aspects = {a["aspect"].lower() for a in attrs}
    for word in query_words:
        if any(word in asp for asp in attr_aspects):
            score += 1
            break

    return min(score, 3)
