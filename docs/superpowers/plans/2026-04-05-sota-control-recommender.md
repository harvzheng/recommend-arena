# SOTA Control Recommender (`design_00_sota`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a state-of-the-art retrieve-and-rerank recommender as a benchmark control, implementing the one-pager's architecture: ABSA extraction → hybrid vector + structured index → query expansion → semantic recall → structured re-rank.

**Architecture:** Ingestion batches reviews per product into single LLM calls for ABSA extraction, normalizes against the product's own attribute schema, and builds both a dense embedding index and structured attribute store. Query pipeline parses the query with one LLM call (extracting weighted attributes, spec constraints, negations, and domain knowledge expansion), retrieves candidates via embedding cosine similarity, then re-ranks using weighted structured attribute matching with negation penalties and spec constraint filtering.

**Tech Stack:** Python, shared LLMProvider (Ollama/Anthropic/OpenAI), numpy for cosine similarity, no external vector DB.

---

## File Structure

```
implementations/design_00_sota/
├── __init__.py           # Factory function (create_recommender)
├── recommender.py        # Main class: ingest() and query() orchestration
├── ingestion.py          # Batch ABSA extraction + attribute normalization
├── index.py              # In-memory product index (embeddings + structured data)
├── query_parser.py       # LLM-based query understanding
├── scorer.py             # Hybrid scoring: semantic recall + structured re-rank
└── prompts.py            # All LLM prompt templates
```

Each file has one job:
- `prompts.py` — all prompt strings, no logic
- `ingestion.py` — calls LLM to extract ABSA tuples, normalizes them
- `index.py` — stores product data, embeddings, and attribute scores; handles lookup
- `query_parser.py` — calls LLM to parse a query into structured form
- `scorer.py` — takes parsed query + index, returns scored/ranked results
- `recommender.py` — wires ingestion → index → query_parser → scorer

---

### Task 1: Prompt Templates

**Files:**
- Create: `implementations/design_00_sota/prompts.py`

- [ ] **Step 1: Create the prompts file**

```python
"""All LLM prompt templates for the SOTA recommender."""

ABSA_EXTRACTION_PROMPT = """\
You are an expert product reviewer analyst. Given a product and its reviews, extract structured attribute assessments.

Product: {product_name} ({category})
Domain: {domain}
Known attributes for this domain: {attribute_names}
Product specs: {specs_json}

Reviews:
{reviews_text}

For each known attribute, analyze the reviews and provide:
- score: float from 0.0 to 10.0 based on reviewer consensus (null if not mentioned)
- confidence: float from 0.0 to 1.0 (how many reviews agree / how clear the signal is)
- snippets: list of 1-3 short direct quotes that support the score

Also extract any additional attributes reviewers mention that aren't in the known list.

Respond with JSON only:
{{
  "attributes": {{
    "attribute_name": {{
      "score": 7.5,
      "confidence": 0.8,
      "snippets": ["quote from review"]
    }}
  }},
  "additional_attributes": {{
    "new_attr_name": {{
      "score": 6.0,
      "confidence": 0.5,
      "snippets": ["quote"]
    }}
  }}
}}"""

QUERY_PARSE_PROMPT = """\
You are a search query analyzer for a {domain} recommendation system.

Available attributes in this domain (1-10 scale): {attribute_names}
Available spec fields: {spec_fields}
Available categories: {categories}

Parse this natural language query into structured search parameters.

Query: "{query_text}"

Instructions:
- Extract desired attributes with importance weights (0.0-1.0)
- Extract spec constraints as field/operator/value triples
- Detect negations ("NOT playful", "not too stiff") as negative attributes
- Expand vague/colloquial terms into concrete attributes (e.g., "ice coast" means high edge_grip and damp for hardpack conditions; "alive underfoot" means high playfulness and responsiveness)
- Identify target categories if mentioned or implied
- For attributes, use ONLY names from the available attributes list above

Respond with JSON only:
{{
  "desired_attributes": [
    {{"name": "stiffness", "weight": 0.9, "direction": "high"}}
  ],
  "negative_attributes": [
    {{"name": "playfulness", "weight": 0.7, "direction": "low"}}
  ],
  "spec_constraints": [
    {{"field": "waist_width_mm", "op": ">=", "value": 105}}
  ],
  "categories": ["freeride"],
  "expanded_terms": ["edge_grip", "hardpack"],
  "query_embedding_text": "a rephrased version of the query using concrete attribute terms for embedding search"
}}"""
```

- [ ] **Step 2: Commit**

```bash
git add implementations/design_00_sota/prompts.py
git commit -m "feat(sota): add LLM prompt templates for ABSA extraction and query parsing"
```

---

### Task 2: Product Index

**Files:**
- Create: `implementations/design_00_sota/index.py`

- [ ] **Step 1: Create the index module**

This is the core data structure — stores everything about products for retrieval and scoring.

```python
"""In-memory product index with embeddings and structured attributes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ProductRecord:
    """All data about a single product for retrieval and scoring."""
    product_id: str
    product_name: str
    domain: str
    category: str
    specs: dict                          # raw spec dict from product data
    ground_truth_attributes: dict        # 1-10 scale from product data
    review_attributes: dict              # ABSA-extracted: {attr: {score, confidence, snippets}}
    embedding: list[float] | None = None # dense vector for semantic search
    review_text_combined: str = ""       # concatenated reviews for embedding


@dataclass
class ProductIndex:
    """In-memory hybrid index: dense embeddings + structured attributes."""

    products: dict[str, ProductRecord] = field(default_factory=dict)
    domain_attributes: dict[str, list[str]] = field(default_factory=dict)
    domain_categories: dict[str, set[str]] = field(default_factory=dict)
    domain_spec_fields: dict[str, list[str]] = field(default_factory=dict)

    def add_product(self, record: ProductRecord) -> None:
        self.products[record.product_id] = record
        domain = record.domain
        if domain not in self.domain_attributes:
            attr_names = [k for k, v in record.ground_truth_attributes.items()
                         if isinstance(v, (int, float))]
            self.domain_attributes[domain] = attr_names
        if domain not in self.domain_categories:
            self.domain_categories[domain] = set()
        self.domain_categories[domain].add(record.category)
        if domain not in self.domain_spec_fields:
            self.domain_spec_fields[domain] = list(record.specs.keys())

    def get_domain_products(self, domain: str) -> list[ProductRecord]:
        return [p for p in self.products.values() if p.domain == domain]

    def get_attribute_names(self, domain: str) -> list[str]:
        return self.domain_attributes.get(domain, [])

    def get_spec_fields(self, domain: str) -> list[str]:
        return self.domain_spec_fields.get(domain, [])

    def get_categories(self, domain: str) -> list[str]:
        return sorted(self.domain_categories.get(domain, set()))

    def clear_domain(self, domain: str) -> None:
        self.products = {pid: p for pid, p in self.products.items()
                         if p.domain != domain}
        self.domain_attributes.pop(domain, None)
        self.domain_categories.pop(domain, None)
        self.domain_spec_fields.pop(domain, None)

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def semantic_search(self, query_embedding: list[float], domain: str,
                        top_k: int) -> list[tuple[ProductRecord, float]]:
        """Return top_k products by cosine similarity to query embedding."""
        candidates = self.get_domain_products(domain)
        scored = []
        for p in candidates:
            if p.embedding is None:
                continue
            sim = self.cosine_similarity(query_embedding, p.embedding)
            scored.append((p, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

- [ ] **Step 2: Commit**

```bash
git add implementations/design_00_sota/index.py
git commit -m "feat(sota): add in-memory product index with embedding search"
```

---

### Task 3: ABSA Ingestion

**Files:**
- Create: `implementations/design_00_sota/ingestion.py`

- [ ] **Step 1: Create the ingestion module**

One LLM call per product (batches all reviews for that product). Falls back to ground-truth attributes if LLM extraction fails.

```python
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
        reviews_text=reviews_text,
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
```

- [ ] **Step 2: Commit**

```bash
git add implementations/design_00_sota/ingestion.py
git commit -m "feat(sota): add batch ABSA extraction with per-product LLM calls"
```

---

### Task 4: Query Parser

**Files:**
- Create: `implementations/design_00_sota/query_parser.py`

- [ ] **Step 1: Create the query parser module**

Single LLM call to decompose a query into structured search parameters.

```python
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
        query_text=query_text,
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
```

- [ ] **Step 2: Commit**

```bash
git add implementations/design_00_sota/query_parser.py
git commit -m "feat(sota): add LLM-based query parser with expansion and negation"
```

---

### Task 5: Hybrid Scorer

**Files:**
- Create: `implementations/design_00_sota/scorer.py`

- [ ] **Step 1: Create the scorer module**

The core ranking logic: semantic recall → structured re-rank with attribute matching, negation penalties, and spec constraint filtering.

```python
"""Hybrid scoring: semantic recall + structured attribute re-ranking."""

from __future__ import annotations

from dataclasses import dataclass, field

from .index import ProductIndex, ProductRecord
from .query_parser import ParsedQuery


@dataclass
class ScoredProduct:
    """A product with its computed scores and explanation data."""
    record: ProductRecord
    semantic_score: float = 0.0
    attribute_score: float = 0.0
    negation_penalty: float = 0.0
    final_score: float = 0.0
    matched_attributes: dict[str, float] = field(default_factory=dict)
    snippets: list[str] = field(default_factory=list)


# Weights for combining signals
SEMANTIC_WEIGHT = 0.3
ATTRIBUTE_WEIGHT = 0.7


def score_and_rank(
    index: ProductIndex,
    parsed_query: ParsedQuery,
    query_embedding: list[float] | None,
    domain: str,
    top_k: int = 10,
) -> list[ScoredProduct]:
    """Score all candidates and return top_k ranked results."""
    candidates = index.get_domain_products(domain)

    # Stage 1: Category filter (soft — keep all if too few match)
    if parsed_query.categories:
        filtered = [p for p in candidates if p.category in parsed_query.categories]
        if len(filtered) >= 2:
            candidates = filtered

    # Stage 2: Spec constraint filter (soft)
    if parsed_query.spec_constraints:
        constrained = [p for p in candidates if _passes_constraints(p, parsed_query.spec_constraints)]
        if len(constrained) >= 2:
            candidates = constrained

    # Stage 3: Score each candidate
    scored = []
    for product in candidates:
        sp = ScoredProduct(record=product)

        # Semantic score from embedding similarity
        if query_embedding and product.embedding:
            sp.semantic_score = index.cosine_similarity(query_embedding, product.embedding)

        # Attribute score from structured data
        sp.attribute_score, sp.matched_attributes, sp.snippets = _compute_attribute_score(
            product, parsed_query
        )

        # Negation penalty
        sp.negation_penalty = _compute_negation_penalty(product, parsed_query)

        # Combine
        sp.final_score = (
            SEMANTIC_WEIGHT * sp.semantic_score
            + ATTRIBUTE_WEIGHT * sp.attribute_score
            - sp.negation_penalty
        )

        scored.append(sp)

    scored.sort(key=lambda x: x.final_score, reverse=True)
    return scored[:top_k]


def _get_attribute_value(product: ProductRecord, attr_name: str) -> float | None:
    """Get attribute value, preferring ABSA-extracted over ground truth.

    Blends review-extracted scores with ground truth when both exist.
    Returns value on 0-10 scale, or None if not found.
    """
    review_data = product.review_attributes.get(attr_name)
    gt_value = product.ground_truth_attributes.get(attr_name)

    if isinstance(gt_value, list):
        gt_value = None  # terrain lists etc. aren't numeric

    if review_data and gt_value is not None:
        review_score = review_data["score"]
        confidence = review_data.get("confidence", 0.5)
        # Blend: higher confidence in reviews → weight reviews more
        return confidence * review_score + (1 - confidence) * float(gt_value)
    elif review_data:
        return review_data["score"]
    elif gt_value is not None:
        return float(gt_value)
    return None


def _compute_attribute_score(
    product: ProductRecord,
    parsed_query: ParsedQuery,
) -> tuple[float, dict[str, float], list[str]]:
    """Compute weighted attribute match score.

    Returns (score, matched_attributes, snippets).
    """
    if not parsed_query.desired_attributes:
        return 0.0, {}, []

    total_weight = 0.0
    weighted_score = 0.0
    matched = {}
    snippets = []

    for attr in parsed_query.desired_attributes:
        name = attr["name"]
        weight = attr["weight"]
        direction = attr.get("direction", "high")
        total_weight += weight

        value = _get_attribute_value(product, name)
        if value is None:
            continue

        # Normalize to 0-1 (from 0-10 scale)
        norm = value / 10.0
        if direction == "low":
            norm = 1.0 - norm

        match_strength = max(0.0, min(1.0, norm))
        weighted_score += weight * match_strength
        matched[name] = round(match_strength, 3)

        # Collect snippets from ABSA data
        review_data = product.review_attributes.get(name)
        if review_data and review_data.get("snippets"):
            snippets.extend(review_data["snippets"][:2])

    if total_weight == 0:
        return 0.0, matched, snippets

    return weighted_score / total_weight, matched, snippets


def _compute_negation_penalty(
    product: ProductRecord,
    parsed_query: ParsedQuery,
) -> float:
    """Compute penalty for attributes the user explicitly doesn't want."""
    if not parsed_query.negative_attributes:
        return 0.0

    total_penalty = 0.0
    for attr in parsed_query.negative_attributes:
        name = attr["name"]
        weight = attr["weight"]

        value = _get_attribute_value(product, name)
        if value is None:
            continue

        # High value on a negated attribute = big penalty
        norm = value / 10.0
        total_penalty += weight * norm * 0.5  # Scale penalty to not overwhelm

    return total_penalty


def _passes_constraints(product: ProductRecord, constraints: list[dict]) -> bool:
    """Check if a product passes all spec constraints."""
    for c in constraints:
        field_name = c["field"]
        op = c["op"]
        target = c["value"]

        if target is None:
            continue

        spec_value = product.specs.get(field_name)
        if spec_value is None:
            continue  # No data — don't penalize

        # Handle list specs (e.g., lengths_cm)
        if isinstance(spec_value, list):
            values = [v for v in spec_value if v is not None]
            if not values:
                continue
            if op == ">=":
                if not any(v >= target for v in values):
                    return False
            elif op == "<=":
                if not any(v <= target for v in values):
                    return False
            elif op == "==":
                if target not in values:
                    return False
        else:
            if op == ">=" and spec_value < target:
                return False
            elif op == "<=" and spec_value > target:
                return False
            elif op == "==" and spec_value != target:
                return False

    return True
```

- [ ] **Step 2: Commit**

```bash
git add implementations/design_00_sota/scorer.py
git commit -m "feat(sota): add hybrid scorer with semantic recall and structured re-rank"
```

---

### Task 6: Main Recommender

**Files:**
- Create: `implementations/design_00_sota/recommender.py`
- Create: `implementations/design_00_sota/__init__.py`

- [ ] **Step 1: Create the recommender module**

Wires everything together: ingestion → index → query_parser → scorer → results.

```python
"""Main Recommender for Design #0: SOTA Control (Retrieve-and-Rerank)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.interface import RecommendationResult
from shared.llm_provider import LLMProvider, get_provider

from .index import ProductIndex
from .ingestion import build_product_record
from .query_parser import parse_query
from .scorer import score_and_rank

logger = logging.getLogger(__name__)


class SotaRecommender:
    """SOTA control: Retrieve-and-Rerank with ABSA extraction.

    Implements the shared Recommender protocol.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm or get_provider()
        self.index = ProductIndex()
        self._ingested_domains: set[str] = set()

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Ingest products and reviews for a domain.

        One LLM call per product (batched reviews) + one embedding per product.
        """
        if domain in self._ingested_domains:
            self.index.clear_domain(domain)
        self._ingested_domains.add(domain)

        # Group reviews by product
        reviews_by_product: dict[str, list[dict]] = {}
        for review in reviews:
            pid = review.get("product_id", "")
            if pid:
                reviews_by_product.setdefault(pid, []).append(review)

        # Infer attribute names from first product
        first_product = products[0] if products else {}
        attribute_names = [
            k for k, v in first_product.get("attributes", {}).items()
            if isinstance(v, (int, float))
        ]

        logger.info("Ingesting %d products for domain '%s' (attributes: %s)",
                     len(products), domain, attribute_names)

        for product in products:
            pid = product.get("product_id") or product.get("id", "")
            product_reviews = reviews_by_product.get(pid, [])

            record = build_product_record(
                product=product,
                reviews=product_reviews,
                domain=domain,
                llm=self.llm,
                attribute_names=attribute_names,
            )

            # Generate embedding
            try:
                record.embedding = self.llm.embed(record.review_text_combined)
            except Exception as e:
                logger.warning("Embedding failed for %s: %s", pid, e)

            self.index.add_product(record)

        logger.info("Ingestion complete: %d products indexed for '%s'",
                     len(products), domain)

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        """Query with natural language, returns ranked recommendations."""
        logger.info("Query: '%s' (domain=%s, top_k=%d)", query_text, domain, top_k)

        # Step 1: Parse query
        parsed = parse_query(
            llm=self.llm,
            query_text=query_text,
            domain=domain,
            attribute_names=self.index.get_attribute_names(domain),
            spec_fields=self.index.get_spec_fields(domain),
            categories=self.index.get_categories(domain),
        )

        logger.info("Parsed: desired=%s, negative=%s, constraints=%s, categories=%s",
                     [a["name"] for a in parsed.desired_attributes],
                     [a["name"] for a in parsed.negative_attributes],
                     parsed.spec_constraints,
                     parsed.categories)

        # Step 2: Embed the query
        query_embedding = None
        try:
            query_embedding = self.llm.embed(parsed.query_embedding_text)
        except Exception as e:
            logger.warning("Query embedding failed: %s", e)

        # Step 3: Score and rank
        scored = score_and_rank(
            index=self.index,
            parsed_query=parsed,
            query_embedding=query_embedding,
            domain=domain,
            top_k=top_k,
        )

        # Step 4: Normalize scores and build results
        if not scored:
            return []

        raw_scores = [sp.final_score for sp in scored]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        score_range = max_s - min_s

        results = []
        for sp in scored:
            if score_range > 0:
                norm_score = (sp.final_score - min_s) / score_range
            else:
                norm_score = 0.5

            norm_score = max(0.0, min(1.0, norm_score))

            explanation = _build_explanation(sp)

            results.append(RecommendationResult(
                product_id=sp.record.product_id,
                product_name=sp.record.product_name,
                score=round(norm_score, 4),
                explanation=explanation,
                matched_attributes=sp.matched_attributes,
            ))

        return results


def _build_explanation(sp) -> str:
    """Build explanation from matched attributes and review snippets."""
    parts = []
    name = sp.record.product_name

    if sp.matched_attributes:
        strong = [a for a, s in sp.matched_attributes.items() if s >= 0.7]
        moderate = [a for a, s in sp.matched_attributes.items() if 0.4 <= s < 0.7]
        weak = [a for a, s in sp.matched_attributes.items() if s < 0.4]

        if strong:
            parts.append(f"{name} excels in: {', '.join(strong)}.")
        if moderate:
            parts.append(f"Solid on: {', '.join(moderate)}.")
        if weak:
            parts.append(f"Weaker on: {', '.join(weak)}.")

    if sp.snippets:
        unique = list(dict.fromkeys(sp.snippets))[:3]
        for snippet in unique:
            parts.append(f'Review: "{snippet}"')

    if not parts:
        parts.append(f"{name} ({sp.record.category}).")

    return " ".join(parts)
```

- [ ] **Step 2: Create the __init__.py factory**

```python
"""Design #0: SOTA Control — Retrieve-and-Rerank Recommender."""

from .recommender import SotaRecommender


def create_recommender() -> SotaRecommender:
    """Factory function used by the benchmark runner."""
    return SotaRecommender()
```

- [ ] **Step 3: Commit**

```bash
git add implementations/design_00_sota/recommender.py implementations/design_00_sota/__init__.py
git commit -m "feat(sota): add main recommender wiring ingestion → index → query → score"
```

---

### Task 7: Smoke Test

**Files:**
- None created; uses existing benchmark infrastructure

- [ ] **Step 1: Verify the module loads**

Run:
```bash
cd /Users/harvey/Development/sports/recommend/benchmark
python -c "
import sys; sys.path.insert(0, '..')
from implementations.design_00_sota import create_recommender
r = create_recommender()
print(f'Loaded: {type(r).__name__}')
print('ingest' in dir(r) and 'query' in dir(r))
"
```

Expected: `Loaded: SotaRecommender` and `True`

- [ ] **Step 2: Run benchmark on design_00_sota only**

Run:
```bash
cd /Users/harvey/Development/sports/recommend/benchmark
python runner.py --recommenders ../implementations --filter "design_00_sota" --runs 1 --output results/design_00
```

Expected: Completes without crashing, produces `results/design_00/summary.txt` and `results/design_00/summary.json`.

- [ ] **Step 3: Fix any runtime errors**

If errors occur, debug and fix. Common issues:
- JSON parsing failures from LLM → already handled with try/except fallbacks
- Embedding dimension mismatches → cosine_similarity handles arbitrary dimensions
- None values in comparisons → _passes_constraints guards against None

- [ ] **Step 4: Commit any fixes**

```bash
git add implementations/design_00_sota/
git commit -m "fix(sota): runtime fixes from smoke test"
```

---

### Task 8: Benchmark and Compare

- [ ] **Step 1: Run the full benchmark comparison**

After design_00_sota passes, compare against existing results.

```bash
cd /Users/harvey/Development/sports/recommend/benchmark
python runner.py --recommenders ../implementations --filter "design_00_sota" --runs 1 --output results/design_00
```

- [ ] **Step 2: Compare results**

Read `results/design_00/summary.json` and compare NDCG@5 against the other 10 designs. The SOTA control should ideally beat the current best (design_05_sql at 0.511 NDCG@5).

- [ ] **Step 3: Commit final results**

```bash
git add benchmark/results/design_00/
git commit -m "bench: add SOTA control benchmark results"
```
