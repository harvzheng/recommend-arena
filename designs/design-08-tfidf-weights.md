# Design 8: TF-IDF + Learned Attribute Weights

## 1. Architecture Overview

This design treats product recommendation as an **information retrieval problem**. Instead of embedding products and queries into a shared dense vector space, we extract structured attributes from unstructured reviews via LLM, represent each product as a sparse feature vector, and score query-product relevance using weighted cosine similarity with TF-IDF-inspired term importance.

The core insight: for product recommendation, the attributes that matter are finite and enumerable within a domain. A ski is stiff or soft, for on-piste or off-piste, has a turn radius, a waist width. These aren't latent dimensions to be discovered -- they're concrete properties that reviewers describe in plain language. Classical IR machinery is well-suited to matching a query's desired attributes against a product's known attributes, especially when we learn which attributes carry the most discriminative weight.

**Why classical ML works here:**
- Product attributes are largely categorical or ordinal, not continuous semantic concepts.
- Users query with explicit attribute preferences ("stiff", "lightweight", "no plate"), not vague vibes.
- Interpretability matters: users want to know *why* a product was recommended.
- The feature space is small enough (dozens to low hundreds of attributes per domain) that sparse methods are efficient and sufficient.
- Speed is trivial -- scoring a few thousand products against a weighted sparse vector is sub-millisecond.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Raw Reviews │────>│ LLM Extract  │────>│ Attribute Vectors │
└──────────────┘     └──────────────┘     └──────────────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  IDF Weights +   │
                                          │  Learned Weights │
                                          └────────┬────────┘
                                                   │
┌──────────────┐     ┌──────────────┐     ┌────────▼────────┐
│  User Query  │────>│ Query Parser │────>│ Weighted Cosine  │
└──────────────┘     └──────────────┘     │ Similarity Score │
                                          └─────────────────┘
```

---

## 2. Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Attribute extraction | Claude API (or local LLM) | Structured extraction from review text |
| Feature engineering | pandas, numpy | Sparse vector construction, normalization |
| Weighting / learning | scikit-learn | TF-IDF transformer, logistic regression for weight learning |
| Storage | SQLite + JSON | Products, attribute vectors, weight matrices |
| Query parsing | Claude API (or regex fallback) | Map natural language to attribute vector |
| Serving | Python (no framework needed for POC) | Direct function calls |

**Total dependencies:** `anthropic` (or `ollama`), `scikit-learn`, `pandas`, `numpy`, `pyyaml`, `sqlite3` (stdlib). No vector database, no embedding model, no GPU.

### 2.1 LLM Provider Abstraction

The system uses an LLM for attribute extraction and query parsing. To support benchmarking with local models and avoid coupling to a single provider, all LLM calls go through a provider abstraction.

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Send a prompt and return the text response."""
        ...

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.content[0].text

class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        import requests
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"num_predict": max_tokens}},
        )
        return response.json()["response"]

# Usage: configure once at startup
llm_provider: LLMProvider = AnthropicProvider()  # or OllamaProvider("llama3")
```

All LLM calls in the extraction and query parsing pipelines use `llm_provider.complete(...)` rather than calling the Anthropic SDK directly. This allows the benchmark harness to swap in Ollama (or any other provider) for consistency across designs.

---

## 3. Data Model

### 3.1 Attribute Catalog

Each domain defines a catalog of known attributes. Attributes have a type that determines how they're encoded into the feature vector.

```python
@dataclass
class AttributeDef:
    name: str                          # e.g. "stiffness"
    attr_type: str                     # "categorical", "ordinal", "numeric", "boolean"
    values: list[str] | None           # e.g. ["soft", "medium", "stiff"] for ordinal
    domain: str                        # e.g. "ski", "running_shoe"

# Example catalog for skis
SKI_CATALOG = [
    AttributeDef("stiffness", "ordinal", ["soft", "medium", "stiff"], "ski"),
    AttributeDef("terrain", "categorical", ["on-piste", "off-piste", "all-mountain", "park"], "ski"),
    AttributeDef("turn_radius", "ordinal", ["short", "medium", "long"], "ski"),
    AttributeDef("weight", "ordinal", ["light", "medium", "heavy"], "ski"),
    AttributeDef("waist_width_mm", "numeric", None, "ski"),
    AttributeDef("length_cm", "numeric", None, "ski"),
    AttributeDef("rocker", "boolean", None, "ski"),
]
```

### 3.2 Product Feature Vector

Each product is represented as a sparse dictionary mapping attribute keys to float values. Categorical attributes are one-hot encoded; ordinal attributes are extracted on a 1-10 scale by the LLM, then normalized to 0.0-1.0; numeric attributes are normalized; booleans are 0/1.

```python
# Example: a stiff on-piste carving ski, 172cm, 68mm waist
{
    "stiffness:stiff": 1.0,
    "stiffness:medium": 0.0,
    "stiffness:soft": 0.0,
    "terrain:on-piste": 0.85,      # confidence from review aggregation
    "terrain:off-piste": 0.1,
    "terrain:all-mountain": 0.3,
    "turn_radius:short": 0.9,
    "waist_width_mm": 0.35,        # normalized: (68 - min) / (max - min)
    "length_cm": 0.6,              # normalized
    "rocker": 0.0,
}
```

### 3.3 Weight Matrix

A per-domain vector of the same dimensionality as the feature vector, storing learned importance weights. Initialized from IDF statistics, then optionally refined from user feedback.

```python
# Stored as JSON per domain
{
    "domain": "ski",
    "idf_weights": {"stiffness:stiff": 1.2, "terrain:on-piste": 0.8, ...},
    "learned_weights": {"stiffness:stiff": 1.5, "terrain:on-piste": 1.1, ...},
    "version": 3,
    "updated_at": "2026-04-05T12:00:00Z"
}
```

### 3.4 SQLite Schema

```sql
CREATE TABLE products (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    name TEXT NOT NULL,
    feature_vector JSON NOT NULL,     -- sparse dict
    metadata JSON,                    -- price, brand, year, etc.
    review_count INTEGER DEFAULT 0
);

CREATE TABLE reviews (
    id TEXT PRIMARY KEY,
    product_id TEXT REFERENCES products(id),
    raw_text TEXT NOT NULL,
    extracted_attributes JSON,        -- LLM extraction result
    source TEXT
);

CREATE TABLE domain_weights (
    domain TEXT PRIMARY KEY,
    idf_weights JSON NOT NULL,
    learned_weights JSON,
    attribute_catalog JSON NOT NULL
);

CREATE INDEX idx_products_domain ON products(domain);
```

---

## 4. Ingestion Pipeline

### Step 1: Extract Attributes via LLM

Each review is passed through an LLM with a domain-specific extraction prompt. The prompt includes the attribute catalog so the LLM maps free text to known attributes.

```python
def extract_attributes(review_text: str, domain: str, catalog: list[AttributeDef]) -> dict:
    prompt = f"""Extract product attributes from this review.
    Domain: {domain}
    Known attributes: {json.dumps([a.__dict__ for a in catalog])}
    Review: {review_text}
    Return JSON with attribute names as keys.
    For ordinal attributes, return an integer from 1-10 indicating intensity
    (e.g., stiffness: 9 for "very stiff", 5 for "medium", 2 for "quite soft").
    For categorical/boolean attributes, use the allowed values.
    Include a _confidence score (0-1) for each extraction."""

    response = llm_provider.complete(prompt=prompt, max_tokens=500)
    return json.loads(response)
```

### Step 2: Aggregate to Product Feature Vector

Multiple reviews for the same product are aggregated. Confidence scores are averaged, and review count serves as a reliability signal.

```python
def aggregate_product_features(extractions: list[dict], catalog: list[AttributeDef]) -> dict:
    feature_vector = defaultdict(list)
    for extraction in extractions:
        for attr_def in catalog:
            if attr_def.name in extraction:
                value = extraction[attr_def.name]
                confidence = extraction.get(f"{attr_def.name}_confidence", 0.7)
                keys = encode_attribute(attr_def, value)
                for k, v in keys.items():
                    feature_vector[k].append(v * confidence)

    # Average across reviews
    return {k: sum(v) / len(v) for k, v in feature_vector.items()}
```

### Step 3: Compute IDF-Style Weights

Standard TF-IDF logic adapted for attribute vectors: attributes that appear in every product (e.g., every ski has a stiffness rating) get lower IDF; rare attributes (e.g., "touring binding compatible") get higher IDF. This naturally emphasizes discriminative attributes.

```python
def compute_idf_weights(all_products: list[dict], feature_keys: list[str]) -> dict:
    n_products = len(all_products)
    idf = {}
    for key in feature_keys:
        # Count products where this attribute is "present" (value > threshold)
        doc_freq = sum(1 for p in all_products if p.get(key, 0) > 0.1)
        if doc_freq == 0:
            idf[key] = 0.0
        else:
            idf[key] = math.log(n_products / doc_freq) + 1.0
    return idf
```

---

## 5. Query Pipeline

### Step 1: Parse Query to Feature Vector

The user's natural language query is converted to a sparse feature vector using the same attribute catalog. The LLM identifies which attributes the user cares about and their desired values. Negations (e.g., "no plate") are encoded as negative weights.

```python
def parse_query(query_text: str, domain: str, catalog: list[AttributeDef]) -> dict:
    prompt = f"""Parse this product query into desired attributes.
    Domain: {domain}
    Known attributes: {json.dumps([a.__dict__ for a in catalog])}
    Query: "{query_text}"
    Return JSON with:
    - desired attributes and target values
    - importance (0-1) for each attribute
    - negations (attributes explicitly NOT wanted)"""

    response = llm_provider.complete(prompt=prompt, max_tokens=300)
    parsed = json.loads(response)
    return build_query_vector(parsed, catalog)

def build_query_vector(parsed: dict, catalog: list[AttributeDef]) -> dict:
    qvec = {}
    for attr_name, info in parsed.get("desired", {}).items():
        attr_def = next(a for a in catalog if a.name == attr_name)
        keys = encode_attribute(attr_def, info["value"])
        importance = info.get("importance", 1.0)
        for k, v in keys.items():
            qvec[k] = v * importance

    for attr_name, info in parsed.get("negations", {}).items():
        attr_def = next(a for a in catalog if a.name == attr_name)
        keys = encode_attribute(attr_def, info["value"])
        for k, v in keys.items():
            qvec[k] = -1.0 * v  # negative signal

    return qvec
```

### Step 2: Weighted Cosine Similarity

Score each product against the query using IDF-weighted (and optionally learned-weighted) cosine similarity.

```python
def score_products(query_vec: dict, products: list[dict], weights: dict) -> list[tuple]:
    scores = []
    for product in products:
        pvec = product["feature_vector"]
        score = weighted_cosine(query_vec, pvec, weights)
        scores.append((product["id"], product["name"], score))

    return sorted(scores, key=lambda x: x[2], reverse=True)

def weighted_cosine(q: dict, p: dict, w: dict) -> float:
    all_keys = set(q.keys()) | set(p.keys())
    dot = sum(q.get(k, 0) * p.get(k, 0) * w.get(k, 1.0) for k in all_keys)
    q_norm = math.sqrt(sum((q.get(k, 0) * w.get(k, 1.0))**2 for k in all_keys))
    p_norm = math.sqrt(sum((p.get(k, 0) * w.get(k, 1.0))**2 for k in all_keys))
    if q_norm == 0 or p_norm == 0:
        return 0.0
    raw = dot / (q_norm * p_norm)
    # Cosine similarity is naturally in [0, 1] for non-negative vectors,
    # but negations in the query vector can push scores negative. Clamp to [0, 1].
    return max(0.0, min(1.0, raw))
```

---

## 6. Ranking Strategy

Ranking uses a three-layer scoring approach:

1. **IDF-weighted cosine similarity** (base signal): Attributes that are rare and discriminative contribute more to the score. A ski being "touring compatible" matters more than it having a stiffness rating (which every ski has).

2. **Hard filter pass**: Numeric constraints (e.g., "180cm+") are applied as filters before scoring, not as soft similarity signals. This avoids the problem of a 160cm ski scoring well because it matches on everything else.

3. **Learned importance weights** (optional refinement): If user feedback is available (clicks, "this was a good recommendation"), train a simple logistic regression or linear model to adjust per-attribute weights. The training signal is: given a (query, product) pair, did the user find it relevant?

```python
def rank(query_text: str, domain: str, db) -> list[dict]:
    catalog = db.get_catalog(domain)
    weights = db.get_weights(domain)

    parsed = parse_query(query_text, domain, catalog)
    query_vec = parsed["feature_vector"]
    hard_filters = parsed["filters"]  # e.g., {"length_cm": {"min": 180}}

    products = db.get_products(domain)
    products = apply_hard_filters(products, hard_filters)

    scored = score_products(query_vec, products, weights)
    return scored[:20]
```

**Tie-breaking:** When similarity scores are close (within 0.05), secondary sort by review count (more reviews = higher confidence in the feature vector) then recency.

---

## 7. Domain Adaptation

Adapting to a new domain requires:

1. **Define an attribute catalog** (~30 minutes of domain knowledge): List the attributes, their types, and possible values. This can itself be bootstrapped by passing a handful of sample reviews to an LLM and asking it to identify the attribute dimensions reviewers discuss.

2. **Define a synonym table** for the domain. This closes the biggest gap with embedding-based approaches: without it, "playful" and "fun" are treated as unrelated features. The synonym table is cheap to maintain and dramatically improves recall.

3. **Run ingestion** on domain-specific review data. IDF weights are computed automatically from the data distribution.

4. **Optionally tune weights** from feedback.

### 7.1 Synonym Tables

Each domain maintains a synonym table in YAML format. Synonyms are grouped by canonical term -- all terms in a group are treated as equivalent during both attribute extraction and query parsing.

```yaml
# synonyms/ski.yaml
synonyms:
  stiff:
    - rigid
    - firm
    - hard
    - demanding
  soft:
    - forgiving
    - flexible
    - buttery
    - easy-going
  playful:
    - fun
    - lively
    - energetic
    - poppy
  stable:
    - planted
    - composed
    - confident
    - locked-in
  light:
    - lightweight
    - featherweight
  heavy:
    - beefy
    - burly
    - tank
  on-piste:
    - groomer
    - corduroy
    - frontside
  off-piste:
    - backcountry
    - powder
    - freeride
  all-mountain:
    - versatile
    - do-it-all
    - quiver-of-one

# synonyms/cookie.yaml
synonyms:
  chewy:
    - gooey
    - soft-baked
    - moist
  crispy:
    - crunchy
    - snappy
    - thin-and-crisp
  sweet:
    - sugary
    - rich
    - decadent
  mild:
    - subtle
    - not-too-sweet
    - lightly-sweetened
```

### 7.2 Synonym Expansion

Before encoding, both query terms and extracted product attributes are expanded through the synonym table. This is a simple dictionary lookup -- no ML required.

```python
def load_synonyms(domain: str) -> dict[str, str]:
    """Load synonym table and build a reverse mapping: synonym -> canonical term."""
    with open(f"synonyms/{domain}.yaml") as f:
        data = yaml.safe_load(f)
    reverse_map = {}
    for canonical, syns in data.get("synonyms", {}).items():
        reverse_map[canonical] = canonical  # canonical maps to itself
        for syn in syns:
            reverse_map[syn.lower()] = canonical
    return reverse_map

def expand_synonyms(attributes: dict, synonym_map: dict[str, str]) -> dict:
    """Replace synonym values with their canonical form in extracted attributes."""
    expanded = {}
    for key, value in attributes.items():
        if isinstance(value, str):
            expanded[key] = synonym_map.get(value.lower(), value)
        else:
            expanded[key] = value
    return expanded
```

Synonym expansion is applied at two points:
1. **During ingestion**, after LLM attribute extraction and before encoding into feature vectors.
2. **During query parsing**, after LLM query parsing and before building the query vector.

This ensures that a query for "fun all-mountain ski" matches products described as "playful" and "versatile" in reviews.

### 7.3 Catalog Bootstrapping

```python
def bootstrap_catalog(sample_reviews: list[str], domain: str) -> list[AttributeDef]:
    """Use LLM to discover attribute dimensions from sample reviews."""
    prompt = f"""Analyze these {domain} reviews and identify the key product
    attributes that reviewers discuss. For each attribute, determine:
    - name, type (categorical/ordinal/numeric/boolean), possible values.
    Reviews: {json.dumps(sample_reviews[:20])}"""

    response = llm_provider.complete(
        prompt=prompt,
        max_tokens=1000,
    )
    return parse_catalog_response(response)
```

The catalog and synonym table are the only domain-specific artifacts. Everything else -- extraction prompts, encoding logic, IDF computation, scoring -- is domain-agnostic. Adding "cookies" as a domain means defining attributes like `texture: [crispy, chewy, cakey]`, `sweetness: [mild, medium, sweet]`, `chocolate_type: [none, milk, dark, white]`, the corresponding synonym table, and feeding in cookie reviews.

---

## 8. Pros and Cons

### Strengths

- **Fully interpretable.** Every recommendation can be explained: "Recommended because: stiffness match (0.95), terrain match (0.88), turn radius match (0.72)." No black-box embeddings.
- **Fast.** Scoring is a dot product over sparse vectors. Thousands of products score in under a millisecond. No ANN index, no GPU.
- **Minimal infrastructure.** SQLite, pandas, scikit-learn. Runs on a laptop. No vector database, no model serving.
- **Negation handling is natural.** "No plate" becomes a negative weight on `plate:true`. Embedding-based systems struggle with negation.
- **Hard constraints are easy.** Numeric filters ("180cm+") are exact, not approximate nearest-neighbor compromises.
- **Debuggable.** When a recommendation is wrong, you can inspect the feature vectors and weights to see exactly why.

### Weaknesses

- **Attribute catalog is a bottleneck.** The system can only match on attributes it knows about. If the catalog misses "vibration dampening" and a user asks for it, the system is blind. (Mitigated by LLM-assisted catalog bootstrapping.)
- **No semantic understanding.** "Playful" and "fun" are different features unless explicitly mapped as synonyms. This is mitigated by the per-domain synonym tables (see Section 7), but embedding approaches handle this more naturally without manual curation.
- **LLM extraction is the weak link.** The quality of feature vectors depends entirely on how well the LLM extracts attributes from messy review text. Hallucinated or missed attributes degrade quality.
- **Scales poorly to very large attribute spaces.** If a domain has 500+ meaningful attributes, the sparse vectors become unwieldy and IDF weights become noisy. Practical limit is ~100-200 attributes per domain.
- **Cold start for weight learning.** IDF weights are a reasonable default, but learned weights require user feedback data that won't exist initially.
- **Cross-domain queries are impossible.** "A ski that feels like my favorite running shoe" cannot be answered because feature spaces don't overlap.

---

## 9. POC Scope

### Goal
Demonstrate end-to-end flow for one domain (skis) with 20-50 products, showing attribute extraction, IDF weighting, and query matching.

### Minimal Implementation

```python
# poc.py -- complete POC sketch

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict

# Uses LLM provider abstraction from Section 2.1
llm_provider = AnthropicProvider()  # or OllamaProvider("llama3")
DB_PATH = "recommend.db"

# --- Catalog ---
SKI_CATALOG = [
    {"name": "stiffness", "type": "ordinal", "values": ["soft", "medium", "stiff"]},
    {"name": "terrain", "type": "categorical",
     "values": ["on-piste", "off-piste", "all-mountain", "park"]},
    {"name": "turn_radius", "type": "ordinal", "values": ["short", "medium", "long"]},
    {"name": "weight", "type": "ordinal", "values": ["light", "medium", "heavy"]},
    {"name": "rocker", "type": "boolean", "values": None},
    {"name": "length_cm", "type": "numeric", "values": None},
    {"name": "waist_width_mm", "type": "numeric", "values": None},
]

def encode_attribute(attr_def: dict, value, confidence: float = 1.0) -> dict:
    """Encode a single attribute value into sparse feature keys."""
    name = attr_def["name"]
    atype = attr_def["type"]

    if atype == "categorical":
        return {f"{name}:{v}": (confidence if v == value else 0.0)
                for v in attr_def["values"]}
    elif atype == "ordinal":
        # LLM extracts ordinal values on a 1-10 scale for finer granularity.
        # This avoids the lossy mapping of "soft/medium/stiff" -> 0.0/0.5/1.0,
        # which can't distinguish "very stiff" from "stiff".
        if isinstance(value, (int, float)):
            normalized = (float(value) - 1.0) / 9.0  # map 1-10 to 0.0-1.0
        else:
            vals = attr_def["values"]
            idx = vals.index(value) if value in vals else len(vals) // 2
            normalized = idx / max(len(vals) - 1, 1)
        return {f"{name}": max(0.0, min(1.0, normalized)) * confidence}
    elif atype == "boolean":
        return {f"{name}": (1.0 if value else 0.0) * confidence}
    elif atype == "numeric":
        return {f"{name}": float(value) * confidence}
    return {}

def extract_review(review_text: str) -> dict:
    """Extract ski attributes from a single review."""
    prompt = f"""Extract ski attributes from this review.
Attributes: {json.dumps(SKI_CATALOG)}
Review: {review_text}
Return JSON only. Use attribute names as keys, values from allowed values.
For ordinal attributes, return an integer from 1-10 indicating intensity.
Add _confidence (0-1) for each."""
    resp = llm_provider.complete(prompt=prompt, max_tokens=400)
    return json.loads(resp)

def compute_idf(products: list[dict]) -> dict:
    """Compute IDF weights across all product feature vectors."""
    n = len(products)
    all_keys = set()
    for p in products:
        all_keys.update(p.keys())

    idf = {}
    for key in all_keys:
        df = sum(1 for p in products if abs(p.get(key, 0)) > 0.1)
        idf[key] = math.log(n / max(df, 1)) + 1.0 if df > 0 else 0.0
    return idf

def weighted_cosine(q: dict, p: dict, w: dict) -> float:
    """Compute weighted cosine similarity between query and product vectors."""
    keys = set(q.keys()) | set(p.keys())
    dot = sum(q.get(k, 0) * p.get(k, 0) * w.get(k, 1.0) for k in keys)
    qn = math.sqrt(sum((q.get(k, 0) * w.get(k, 1.0)) ** 2 for k in keys))
    pn = math.sqrt(sum((p.get(k, 0) * w.get(k, 1.0)) ** 2 for k in keys))
    if not qn or not pn:
        return 0.0
    # Clamp to [0, 1]: non-negative vectors give [0, 1] naturally, but
    # negations in the query can push scores negative.
    return max(0.0, min(1.0, dot / (qn * pn)))

def query(query_text: str, products: list[dict], idf: dict) -> list:
    """Parse a natural language query and rank products."""
    prompt = f"""Parse this ski query into attributes.
Catalog: {json.dumps(SKI_CATALOG)}
Query: "{query_text}"
Return JSON with "desired" (attr->value mapping) and "negations" (attr->value)."""
    resp = llm_provider.complete(prompt=prompt, max_tokens=300)
    parsed = json.loads(resp)

    # Build query vector
    qvec = {}
    for attr_name, value in parsed.get("desired", {}).items():
        attr_def = next((a for a in SKI_CATALOG if a["name"] == attr_name), None)
        if attr_def:
            qvec.update(encode_attribute(attr_def, value))
    for attr_name, value in parsed.get("negations", {}).items():
        attr_def = next((a for a in SKI_CATALOG if a["name"] == attr_name), None)
        if attr_def:
            for k, v in encode_attribute(attr_def, value).items():
                qvec[k] = -v

    # Score and rank
    results = []
    for name, fvec in products:
        score = weighted_cosine(qvec, fvec, idf)
        results.append((name, score, explain_score(qvec, fvec, idf)))
    return sorted(results, key=lambda x: x[1], reverse=True)

def explain_score(q: dict, p: dict, w: dict) -> str:
    """Generate a human-readable explanation string for a score.

    Returns a string like:
      "Matched on stiffness (high match, +0.45), terrain: on-piste (+0.32).
       Penalized for rocker (-0.15)."
    """
    positives = []
    negatives = []
    for k in set(q.keys()) & set(p.keys()):
        contrib = q[k] * p[k] * w.get(k, 1.0)
        if contrib > 0.01:
            strength = "high" if contrib > 0.3 else "moderate" if contrib > 0.15 else "slight"
            positives.append((k, contrib, strength))
        elif contrib < -0.01:
            negatives.append((k, contrib))

    positives.sort(key=lambda x: x[1], reverse=True)
    negatives.sort(key=lambda x: x[1])

    parts = []
    if positives:
        matched = ", ".join(f"{k} ({strength} match, {c:+.2f})" for k, c, strength in positives)
        parts.append(f"Matched on {matched}")
    if negatives:
        penalized = ", ".join(f"{k} ({c:+.2f})" for k, c in negatives)
        parts.append(f"Penalized for {penalized}")

    return ". ".join(parts) + "." if parts else "No significant attribute overlap."

# --- Entry point ---
if __name__ == "__main__":
    # Simulated product vectors (in practice, built from review extraction)
    demo_products = [
        ("Nordica Dobermann", {"stiffness": 1.0, "terrain:on-piste": 0.95,
                               "terrain:off-piste": 0.05, "turn_radius": 0.2,
                               "weight": 0.8, "rocker": 0.0}),
        ("Blizzard Rustler 10", {"stiffness": 0.5, "terrain:on-piste": 0.3,
                                  "terrain:off-piste": 0.8, "turn_radius": 0.6,
                                  "weight": 0.5, "rocker": 1.0}),
    ]
    idf_weights = compute_idf([p[1] for p in demo_products])

    results = query("stiff on-piste carving ski, short turns", demo_products, idf_weights)
    for name, score, explanation in results:
        print(f"{name}: {score:.3f}")
        print(f"  {explanation}")
```

### POC Milestones

1. **Catalog definition + encoding** (2 hours): Define ski attribute catalog, implement encode/decode, write unit tests for encoding round-trips.
2. **LLM extraction** (3 hours): Prompt engineering for reliable attribute extraction from reviews, batch process 50 sample reviews, measure extraction accuracy.
3. **IDF computation + scoring** (2 hours): Compute IDF from product corpus, implement weighted cosine, verify ranking on known-good queries.
4. **Query parsing + end-to-end demo** (3 hours): LLM-based query parsing, explanation generation, interactive CLI demo.
5. **Evaluation** (2 hours): Test with 10-15 queries, compare rankings against human judgment, identify failure modes.

**Total estimated POC effort: ~12 hours.**

### Success Criteria

- Given "stiff, on-piste carving ski," the system ranks known carving skis (Dobermann, Redster, etc.) in the top 3.
- Given "playful all-mountain ski, light," the system ranks all-mountain skis above race skis.
- Negation works: "no rocker" penalizes rockered skis.
- Each recommendation includes a human-readable explanation of why it ranked where it did.

---

## 10. Common Interface

All designs in the benchmark implement a common protocol so they can be evaluated uniformly. This design's implementation of the shared interface maps the ingestion and query flows described above onto the standard `Recommender` contract.

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class RecommendationResult:
    product_id: str
    product_name: str
    score: float  # 0-1 normalized
    explanation: str
    matched_attributes: dict[str, float]

class Recommender(Protocol):
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None: ...
    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]: ...
```

### 10.1 Implementation

```python
class TFIDFRecommender:
    """Recommender implementation using TF-IDF weighted attribute vectors."""

    def __init__(self, llm: LLMProvider, synonyms_dir: str = "synonyms"):
        self.llm = llm
        self.synonyms_dir = synonyms_dir
        self.catalogs: dict[str, list[dict]] = {}
        self.products: dict[str, list[tuple[str, str, dict]]] = {}  # domain -> [(id, name, fvec)]
        self.idf_weights: dict[str, dict] = {}
        self.synonym_maps: dict[str, dict[str, str]] = {}

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        catalog = self._get_or_bootstrap_catalog(domain, reviews)
        self.catalogs[domain] = catalog
        self.synonym_maps[domain] = load_synonyms(domain)
        syn_map = self.synonym_maps[domain]

        # Group reviews by product
        reviews_by_product = defaultdict(list)
        for review in reviews:
            reviews_by_product[review["product_id"]].append(review["text"])

        # Extract and aggregate
        product_vectors = []
        for product in products:
            pid = product["id"]
            extractions = []
            for review_text in reviews_by_product.get(pid, []):
                attrs = extract_attributes(review_text, domain, catalog)
                attrs = expand_synonyms(attrs, syn_map)
                extractions.append(attrs)
            fvec = aggregate_product_features(extractions, catalog) if extractions else {}
            product_vectors.append((pid, product["name"], fvec))

        self.products[domain] = product_vectors
        self.idf_weights[domain] = compute_idf_weights(
            [pv[2] for pv in product_vectors],
            list({k for _, _, fv in product_vectors for k in fv.keys()}),
        )

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        catalog = self.catalogs[domain]
        weights = self.idf_weights[domain]
        syn_map = self.synonym_maps.get(domain, {})

        parsed = parse_query(query_text, domain, catalog)
        parsed = expand_synonyms(parsed, syn_map)
        query_vec = build_query_vector(parsed, catalog)

        results = []
        for pid, pname, pvec in self.products.get(domain, []):
            raw_score = weighted_cosine(query_vec, pvec, weights)
            score = max(0.0, min(1.0, raw_score))

            # Build matched_attributes: per-attribute contribution
            matched = {}
            for k in set(query_vec.keys()) & set(pvec.keys()):
                contrib = query_vec[k] * pvec[k] * weights.get(k, 1.0)
                if abs(contrib) > 0.01:
                    matched[k] = round(contrib, 3)

            explanation = explain_score(query_vec, pvec, weights)

            results.append(RecommendationResult(
                product_id=pid,
                product_name=pname,
                score=round(score, 4),
                explanation=explanation,
                matched_attributes=matched,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
```

The `TFIDFRecommender` class wires together all the components described in this design: LLM provider abstraction (Section 2.1), synonym expansion (Section 7.2), attribute extraction (Section 4), IDF weighting (Section 4), weighted cosine scoring with [0, 1] clamping (Section 5), and human-readable explanations (Section 9).
