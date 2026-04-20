# Design 05: SQL-First / SQLite + Full-Text Search

## 1. Architecture Overview

This design bets on a claim: **most recommendation queries are structured queries in disguise.** When a user says "stiff, on-piste carving ski, 180cm+," they are expressing a conjunction of attribute filters with implicit weighting -- exactly what relational databases were built for.

The architecture is deliberately simple. SQLite is the single data store: structured attribute tables for precise filtering, FTS5 for free-text matching against review content, and a thin Python layer that stitches results together. An LLM is used in exactly two places -- ingestion (extracting attributes from reviews) and query time (parsing natural language into structured filters) -- and nowhere else. No vector store, no embeddings, no retrieval-augmented generation at query time.

Why simplicity wins here:

- **Deterministic ranking.** BM25 scores and attribute matches are explainable and debuggable. You can inspect every factor that influenced a recommendation.
- **Zero infrastructure.** A single `.db` file. No server processes, no GPU, no API calls at query time (after the one-time parse).
- **Sub-millisecond queries.** SQLite FTS5 on datasets of this scale (thousands to low millions of reviews) returns in under 10ms.
- **Portability.** The database is a file. Ship it, copy it, back it up with `cp`.

```
                +-----------+
                |   User    |
                +-----+-----+
                      |  natural language query
                      v
              +-------+--------+
              | LLM Query Parse|  (single LLM call)
              +-------+--------+
                      |  structured filters + keywords
                      v
        +-------------+-------------+
        |        SQLite Engine      |
        |  +------+  +-----------+  |
        |  | FTS5 |  | Attribute |  |
        |  |Index |  |  Tables   |  |
        |  +------+  +-----------+  |
        +-------------+-------------+
                      |
                      v
              +-------+--------+
              | Rank & Merge   |  (Python, no LLM)
              +-------+--------+
                      |
                      v
               Ranked results
```

## 2. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Database | SQLite 3.40+ (with FTS5) | Built into Python stdlib, zero setup |
| Language | Python 3.11+ | `sqlite3` in stdlib, broad ecosystem |
| LLM | Pluggable provider (OpenAI, Ollama, etc.) | Used only for extraction and query parsing; see section 12 |
| Schema migrations | Hand-written SQL | No ORM overhead for a single-file DB |
| CLI / API | Click or FastAPI (optional) | Thin interface layer |

Total external dependencies: one LLM client library. That is the entire `requirements.txt`.

## 3. Data Model

```sql
-- Core entities
CREATE TABLE domains (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,          -- 'skis', 'running_shoes', 'cookies'
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domains(id),
    name TEXT NOT NULL,
    brand TEXT,
    year INTEGER,
    raw_specs TEXT,                      -- JSON blob of original specs
    created_at TEXT DEFAULT (datetime('now'))
);

-- Attribute system (domain-specific, schema-driven)
CREATE TABLE attribute_defs (
    id INTEGER PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domains(id),
    name TEXT NOT NULL,                  -- 'stiffness', 'terrain', 'length_cm'
    data_type TEXT NOT NULL,             -- 'numeric', 'categorical', 'boolean', 'scale'
    scale_min REAL,                      -- for scale type: e.g., 1
    scale_max REAL,                      -- for scale type: e.g., 10
    allowed_values TEXT,                 -- JSON array for categorical: ["on-piste","off-piste","all-mountain"]
    UNIQUE(domain_id, name)
);

CREATE TABLE product_attributes (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    attribute_def_id INTEGER NOT NULL REFERENCES attribute_defs(id),
    value_numeric REAL,
    value_text TEXT,
    confidence REAL DEFAULT 1.0,        -- LLM extraction confidence
    source TEXT,                         -- 'review_aggregate', 'spec_sheet', 'llm_extracted'
    UNIQUE(product_id, attribute_def_id)
);

-- Reviews and full-text search
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    source TEXT,                         -- 'reddit', 'blister', 'rei'
    author TEXT,
    content TEXT NOT NULL,
    rating REAL,
    created_at TEXT
);

-- FTS5 virtual table for review content
CREATE VIRTUAL TABLE reviews_fts USING fts5(
    content,
    product_name,                        -- denormalized for search
    content='reviews',
    content_rowid='id',
    tokenize='porter unicode61'          -- stemming + unicode support
);

-- FTS5 triggers to keep index in sync
CREATE TRIGGER reviews_ai AFTER INSERT ON reviews BEGIN
    INSERT INTO reviews_fts(rowid, content, product_name)
    SELECT new.id, new.content, p.name FROM products p WHERE p.id = new.product_id;
END;

CREATE TRIGGER reviews_ad AFTER DELETE ON reviews BEGIN
    INSERT INTO reviews_fts(reviews_fts, rowid, content, product_name)
    VALUES('delete', old.id, old.content,
           (SELECT name FROM products WHERE id = old.product_id));
END;
```

Key design decisions:

- **Attribute values are per-product, not per-review.** During ingestion, we aggregate across reviews to produce a single consensus value (e.g., average stiffness = 8.2/10). The `confidence` field tracks how consistent reviewers were.
- **`data_type` drives query behavior.** Numeric attributes support range queries (`length_cm >= 180`). Categorical attributes support `IN` clauses. Scale attributes support fuzzy matching with distance penalties.
- **FTS5 with Porter stemming** handles morphological variation ("responsive" matches "responsiveness") without embeddings.

## 4. Ingestion Pipeline

```
Raw review text
      |
      v
+-----+---------+
| LLM Extraction |  "Extract attributes from this ski review.
|   (batched)    |   Domain: skis. Attributes: stiffness(1-10),
+-----+---------+   terrain(categorical), turn_radius(numeric)..."
      |
      v
  Structured JSON per review:
  { "stiffness": 8, "terrain": ["on-piste", "groomer"],
    "turn_radius_m": 14, "length_mentioned": 182 }
      |
      v
+-----+---------+
| Aggregation    |  Per product: median stiffness across reviews,
| & Consensus    |  mode terrain, confidence = 1/stddev
+-----+---------+
      |
      v
  INSERT into products, product_attributes, reviews
  (FTS index auto-updates via trigger)
```

The LLM extraction prompt is templated per domain. For skis:

```python
EXTRACTION_PROMPT = """
Extract structured attributes from this ski review.
Return JSON with these fields (omit if not mentioned):
- stiffness: integer 1-10 (1=soft, 10=race-stiff)
- terrain: list from [on-piste, off-piste, all-mountain, park, touring]
- turn_radius: meters (numeric)
- skill_level: list from [beginner, intermediate, advanced, expert]
- snow_conditions: list from [groomed, crud, powder, ice, variable]
- sentiment: float -1.0 to 1.0

Review: {review_text}
"""
```

Extraction is batched and cached. A review is processed once; re-extraction only happens if attribute definitions change.

## 5. Query Pipeline

A user query like **"stiff, on-piste carving ski, 180cm+"** is processed in three stages.

**Stage 1: LLM Query Parse (single call)**

```python
QUERY_PARSE_PROMPT = """
Parse this product query into structured filters and free-text keywords.
Domain: {domain}. Available attributes: {attribute_defs_json}

Query: "{user_query}"

Return JSON:
{
  "filters": [
    {"attribute": "stiffness", "op": "gte", "value": 7},
    {"attribute": "terrain", "op": "contains", "value": "on-piste"},
    {"attribute": "length_cm", "op": "gte", "value": 180}
  ],
  "keywords": "carving",
  "sort_preference": "stiffness_desc"
}
"""
```

**Stage 2: SQL Generation (deterministic, no LLM)**

The parsed filters are converted to parameterized SQL. This is template-driven, not LLM-generated -- no prompt injection risk.

```python
def build_query(parsed):
    conditions = []
    params = []
    joins = []

    for i, f in enumerate(parsed["filters"]):
        alias = f"pa{i}"
        joins.append(
            f"JOIN product_attributes {alias} ON {alias}.product_id = p.id "
            f"AND {alias}.attribute_def_id = ?"
        )
        params.append(get_attr_def_id(f["attribute"]))

        op_map = {"gte": ">=", "lte": "<=", "eq": "="}
        if f["op"] == "contains":
            conditions.append(f"{alias}.value_text LIKE ?")
            params.append(f"%{f['value']}%")
        else:
            conditions.append(f"{alias}.value_numeric {op_map[f['op']]} ?")
            params.append(f["value"])

    # FTS component for keywords
    fts_clause = ""
    if parsed.get("keywords"):
        fts_clause = """
            JOIN (
                SELECT rowid, rank as fts_rank
                FROM reviews_fts WHERE reviews_fts MATCH ?
            ) fts ON fts.rowid IN (
                SELECT id FROM reviews WHERE product_id = p.id
            )
        """
        params.append(parsed["keywords"])

    where = " AND ".join(conditions) if conditions else "1=1"

    sql = f"""
        SELECT p.id, p.name, p.brand
        FROM products p
        {' '.join(joins)}
        {fts_clause}
        WHERE {where}
        GROUP BY p.id
    """
    return sql, params
```

**Stage 3: Rank and Return**

Results from the SQL query are scored and sorted in Python (see section 6).

## 6. Ranking Strategy

The final score for each product is a weighted combination of three signals:

```
score = w1 * attribute_match_score + w2 * fts_bm25_score + w3 * review_sentiment_score
```

**Attribute match score (0-1):** For each filter, compute how well the product matches. Exact matches score 1.0. For scale/numeric attributes, apply a decay function based on distance from the target value:

```python
def attribute_score(actual, target, op, scale_range):
    if op in ("gte", "lte"):
        if meets_condition(actual, target, op):
            return 1.0
        distance = abs(actual - target) / scale_range
        return max(0, 1.0 - distance)  # linear decay for near-misses
    elif op == "eq":
        return 1.0 if actual == target else 0.0
```

Near-misses are not discarded -- a 178cm ski still appears for a "180cm+" query, just ranked lower. This is a key differentiator: rigid filter-based systems would silently drop a 178cm ski that is otherwise a perfect match. The linear decay makes the penalty proportional and transparent. The POC demo output explicitly annotates near-miss results (e.g., `[NEAR-MISS: length_cm=178, target=180+, penalty=-0.02]`) so the benchmark evaluator can verify this behavior.

**FTS BM25 score:** SQLite FTS5 provides `bm25()` natively. Raw BM25 scores are unbounded negative values (more negative = better match in FTS5's convention), so they must be normalized to 0-1 for the composite score. We use min-max normalization across the result set:

```python
def normalize_bm25_scores(raw_scores: list[float]) -> list[float]:
    """Normalize FTS5 BM25 scores to 0-1 range.

    FTS5 bm25() returns negative values where more negative = better match.
    We negate first so higher = better, then apply min-max normalization.
    Falls back to sigmoid normalization when the result set has fewer than
    2 distinct scores (avoids division by zero).
    """
    if not raw_scores:
        return []
    negated = [-s for s in raw_scores]  # flip so higher = better
    lo, hi = min(negated), max(negated)
    if hi - lo < 1e-9:
        # Single-value result set: use sigmoid as fallback
        import math
        return [1.0 / (1.0 + math.exp(-s)) for s in negated]
    return [(s - lo) / (hi - lo) for s in negated]
```

Min-max is preferred because it is simple, deterministic, and spreads scores across the full 0-1 range within each query. The downside is that scores are not comparable across queries -- but that is acceptable since we never compare results from different queries. For workloads that need cross-query stability, a sigmoid transform (`1 / (1 + exp(-k * score))` with tunable `k`) is the alternative.

This captures how well review text matches the free-text portion of the query ("carving" appearing frequently in reviews is a strong signal).

**Review sentiment score (0-1):** Pre-computed during ingestion. Average sentiment across all reviews for the product, shifted to 0-1 range.

Default weights: `w1=0.5, w2=0.35, w3=0.15`. These are configurable per domain and tunable against relevance judgments.

## 7. Domain Adaptation

Adding a new domain requires zero code changes. The process:

1. **Define the domain** -- insert a row into `domains`.
2. **Define attributes** -- insert rows into `attribute_defs` with names, types, and constraints.
3. **Provide an extraction prompt template** -- stored as a config file or in a `domain_configs` table.
4. **Ingest data** -- the pipeline reads attribute definitions and generates extraction prompts dynamically.

```python
# Adding "running_shoes" domain
db.execute("INSERT INTO domains (name) VALUES ('running_shoes')")
domain_id = db.lastrowid

attributes = [
    ("cushioning", "scale", 1, 10, None),
    ("weight_g", "numeric", None, None, None),
    ("drop_mm", "numeric", None, None, None),
    ("has_plate", "boolean", None, None, None),
    ("terrain", "categorical", None, None,
     '["road","trail","track","treadmill"]'),
    ("arch_support", "scale", 1, 10, None),
]
for name, dtype, smin, smax, allowed in attributes:
    db.execute(
        "INSERT INTO attribute_defs (domain_id, name, data_type, "
        "scale_min, scale_max, allowed_values) VALUES (?,?,?,?,?,?)",
        (domain_id, name, dtype, smin, smax, allowed)
    )
```

The query parser prompt automatically adapts because it receives the current attribute definitions as context. No retraining, no new embeddings, no schema migration.

## 8. Pros and Cons

### Strengths

- **Radical simplicity.** One file, one process, one language. A junior developer can understand and debug the entire system in an afternoon.
- **Deterministic and explainable.** Every recommendation can be traced to specific attribute matches and BM25 scores. No black-box embedding similarity.
- **Fast.** Query latency is dominated by the single LLM call for query parsing (~500ms). The SQL query itself runs in microseconds.
- **No cold start for attributes.** Once attribute definitions exist, the system works with even a single review per product.
- **Auditable.** The `confidence` field and `source` tracking on attributes make it easy to spot extraction errors.

### Weaknesses

- **Semantic gap.** BM25 cannot understand that "forgiving" and "easy to ski" mean the same thing. This is mitigated by a per-domain synonym expansion table (see section 11) that maps user terms to review terms before FTS queries, but edge cases remain -- embeddings handle truly open-ended semantic similarity better.
- **LLM dependency for quality.** Attribute extraction quality is the ceiling for recommendation quality. If the LLM misclassifies stiffness, no amount of clever SQL fixes it.
- **Attribute completeness.** The system only knows what you define. If users care about an attribute you did not model (say, "vibration dampening"), it falls through to BM25 keyword matching, which is weaker.
- **No learning from user behavior.** There is no feedback loop. The system does not get better as users interact with it (unlike collaborative filtering or embedding fine-tuning).
- **Cross-attribute reasoning is manual.** "A ski that's good for aggressive intermediates" requires the system to know that maps to stiffness 6-8 + all-mountain terrain. The LLM query parser handles this, but it is doing heavy lifting.

### Honest assessment

This design is the right choice when: data volume is moderate (<1M reviews), attributes are well-defined, explainability matters, and operational simplicity is a priority. It is the wrong choice when: semantic similarity is the primary signal, the domain has poorly-defined attributes, or the system needs to handle highly ambiguous queries where keyword matching fails.

## 9. POC Scope

A minimal proof-of-concept covering skis with ~100 reviews.

```python
# poc.py -- complete sketch, ~200 lines when filled out
import sqlite3
import json
from recommend.llm_provider import get_llm_provider  # see section 12

DB_PATH = "recommend.db"
llm = get_llm_provider()  # configurable: OpenAI, Ollama, etc.

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA_SQL)  # full schema from section 3
    # seed ski domain + attribute defs
    return db

def extract_attributes(review_text: str, domain: str, attr_defs: list) -> dict:
    """Single LLM call to extract structured attributes from a review."""
    prompt = EXTRACTION_PROMPT.format(
        review_text=review_text,
        attributes=json.dumps(attr_defs)
    )
    return json.loads(llm(prompt, json_mode=True))

def ingest_review(db, product_id: int, review_text: str, attr_defs: list):
    """Extract attributes and insert review + attributes."""
    attrs = extract_attributes(review_text, "skis", attr_defs)
    db.execute(
        "INSERT INTO reviews (product_id, content) VALUES (?, ?)",
        (product_id, review_text)
    )
    for attr_name, value in attrs.items():
        # upsert into product_attributes with running average
        update_product_attribute(db, product_id, attr_name, value)
    db.commit()

def parse_query(user_query: str, domain: str, attr_defs: list) -> dict:
    """Single LLM call to decompose query into filters + keywords."""
    prompt = QUERY_PARSE_PROMPT.format(
        domain=domain,
        attribute_defs_json=json.dumps(attr_defs),
        user_query=user_query
    )
    return json.loads(llm(prompt, json_mode=True))

def recommend(db, user_query: str, domain: str = "skis", top_k: int = 5):
    """Full query pipeline: parse -> SQL -> rank -> return."""
    attr_defs = get_attribute_defs(db, domain)
    parsed = parse_query(user_query, domain, attr_defs)

    # Expand keywords with synonyms before FTS query (see section 11)
    if parsed.get("keywords"):
        parsed["keywords"] = expand_synonyms(parsed["keywords"], domain)

    sql, params = build_query(parsed)
    candidates = db.execute(sql, params).fetchall()

    scored = []
    for row in candidates:
        attr_scores = compute_attribute_scores(db, row["id"], parsed["filters"])
        fts_score = compute_fts_score(db, row["id"], parsed.get("keywords", ""))
        sent_score = get_sentiment_score(db, row["id"])
        attr_score = sum(attr_scores.values()) / len(attr_scores) if attr_scores else 0
        total = 0.50 * attr_score + 0.35 * fts_score + 0.15 * sent_score
        explanation = build_explanation(attr_scores, fts_score, sent_score, total)
        scored.append((row, total, attr_scores, fts_score, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

if __name__ == "__main__":
    db = init_db()
    # ingest sample data...
    results = recommend(db, "stiff on-piste carving ski, 180cm+")
    for product, score, attr_scores, fts_s, explanation in results:
        print(f"\n{'='*60}")
        print(f"  {product['name']}  --  score: {score:.3f}")
        print(explanation)
```

**POC deliverables:**

1. Working SQLite database with schema, ski domain, and 5-10 attribute definitions
2. Ingestion of ~100 ski reviews from a seed dataset (JSON file)
3. End-to-end query flow for 3-5 sample queries
4. Printed ranking with per-attribute score breakdowns and near-miss annotations (see section 13)
5. Demonstration of adding a second domain (running shoes) with zero code changes
6. Synonym expansion config for the ski domain (`config/synonyms/skis.yaml`)
7. Implementation of the `Recommender` protocol (section 10) for benchmark compatibility

**Estimated effort:** 1-2 days for a working POC, assuming reviews are already collected.

## 10. Common Interface

All recommendation designs implement a common interface so they can be benchmarked interchangeably. This design conforms to the shared `Recommender` protocol:

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

The SQL-first implementation:

```python
class SqlRecommender:
    """Recommender backed by SQLite + FTS5."""

    def __init__(self, db_path: str = ":memory:", llm_provider=None):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA_SQL)
        self.llm = llm_provider or get_llm_provider()

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        ensure_domain(self.db, domain)
        for product in products:
            insert_product(self.db, product, domain)
        for review in reviews:
            ingest_review(self.db, review["product_id"], review["content"],
                          get_attribute_defs(self.db, domain))
        self.db.commit()

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        attr_defs = get_attribute_defs(self.db, domain)
        parsed = parse_query(query_text, domain, attr_defs)
        if parsed.get("keywords"):
            parsed["keywords"] = expand_synonyms(parsed["keywords"], domain)

        sql, params = build_query(parsed)
        candidates = self.db.execute(sql, params).fetchall()

        scored = []
        for row in candidates:
            attr_scores = compute_attribute_scores(self.db, row["id"], parsed["filters"])
            fts_score = compute_fts_score(self.db, row["id"], parsed.get("keywords", ""))
            sent_score = get_sentiment_score(self.db, row["id"])
            attr_score = sum(attr_scores.values()) / len(attr_scores) if attr_scores else 0
            total = 0.50 * attr_score + 0.35 * fts_score + 0.15 * sent_score
            explanation = build_explanation(attr_scores, fts_score, sent_score, total)
            scored.append(RecommendationResult(
                product_id=str(row["id"]),
                product_name=row["name"],
                score=total,
                explanation=explanation,
                matched_attributes=attr_scores,
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
```

## 11. Synonym Expansion

The semantic gap (BM25 not understanding that "forgiving" and "easy to ski" mean the same thing) is mitigated with a per-domain synonym expansion table. Synonyms are loaded from a YAML config file, consistent with how other designs handle domain configuration.

```yaml
# config/synonyms/skis.yaml
synonyms:
  forgiving:
    - "easy to ski"
    - "forgiving"
    - "beginner-friendly"
    - "user-friendly"
  responsive:
    - "responsive"
    - "quick edge to edge"
    - "snappy"
    - "lively"
  stable:
    - "stable"
    - "damp"
    - "composed"
    - "planted"
  playful:
    - "playful"
    - "fun"
    - "loose"
    - "surfy"
  charger:
    - "charger"
    - "go fast"
    - "high speed"
    - "lay into it"
```

At query time, keywords from the LLM parse are expanded before being sent to FTS5:

```python
import yaml
from pathlib import Path

_synonym_cache: dict[str, dict[str, list[str]]] = {}

def load_synonyms(domain: str) -> dict[str, list[str]]:
    if domain not in _synonym_cache:
        path = Path(f"config/synonyms/{domain}.yaml")
        if path.exists():
            data = yaml.safe_load(path.read_text())
            _synonym_cache[domain] = data.get("synonyms", {})
        else:
            _synonym_cache[domain] = {}
    return _synonym_cache[domain]

def expand_synonyms(keywords: str, domain: str) -> str:
    """Expand query keywords using the synonym table.

    For each keyword that has synonyms, builds an FTS5 OR query so
    that any synonym variant matches. Terms without synonyms are
    passed through unchanged.
    """
    synonyms = load_synonyms(domain)
    tokens = keywords.lower().split()
    expanded_parts = []
    for token in tokens:
        if token in synonyms:
            # Build FTS5 OR group: ("easy to ski" OR "forgiving" OR "beginner-friendly")
            alternatives = " OR ".join(f'"{s}"' for s in synonyms[token])
            expanded_parts.append(f"({alternatives})")
        else:
            expanded_parts.append(token)
    return " ".join(expanded_parts)
```

This is cheap (a dictionary lookup per keyword), requires no embeddings, and can be extended by domain experts editing a YAML file. It does not solve the general semantic similarity problem, but it closes the gap for the most common and important term mappings.

## 12. LLM Provider Abstraction

The POC uses a simple callable interface for LLM access, allowing backends to be swapped without changing application code:

```python
from typing import Callable

# Type alias: takes a prompt string and returns a response string.
# Optional json_mode flag requests JSON-formatted output.
LlmProvider = Callable[..., str]

def openai_provider(model: str = "gpt-4o-mini") -> LlmProvider:
    """Provider backed by the OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    def call(prompt: str, json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content
    return call

def ollama_provider(model: str = "llama3.1") -> LlmProvider:
    """Provider backed by a local Ollama instance."""
    import requests

    def call(prompt: str, json_mode: bool = False) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
        resp = requests.post("http://localhost:11434/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]
    return call

def get_llm_provider(backend: str | None = None, model: str | None = None) -> LlmProvider:
    """Factory that reads from env or explicit args.

    Set LLM_BACKEND=ollama and LLM_MODEL=llama3.1 for local inference,
    or LLM_BACKEND=openai (default) for API-based inference.
    """
    import os
    backend = backend or os.environ.get("LLM_BACKEND", "openai")
    if backend == "ollama":
        return ollama_provider(model or os.environ.get("LLM_MODEL", "llama3.1"))
    else:
        return openai_provider(model or os.environ.get("LLM_MODEL", "gpt-4o-mini"))
```

Adding a new provider (Anthropic, Azure, a mock for testing) means writing one function that takes a prompt string and returns a response string.

## 13. Explanation Generation

Every recommendation includes a human-readable explanation built from score components. This serves both the end user (understanding why a product was recommended) and the benchmark evaluator (verifying that scoring works correctly).

```python
def build_explanation(
    attr_scores: dict[str, float],
    fts_score: float,
    sent_score: float,
    total_score: float,
) -> str:
    """Build explanation string from individual score components.

    Example output:
        Score: 0.847
        Attribute breakdown:
          stiffness:  1.00  (exact match)
          terrain:    1.00  (exact match)
          length_cm:  0.80  [NEAR-MISS: actual=178, target=180+, penalty=-0.20]
        FTS relevance:  0.72
        Sentiment:      0.91
        Weighted total: 0.50*0.93 + 0.35*0.72 + 0.15*0.91 = 0.847
    """
    lines = [f"  Score: {total_score:.3f}", "  Attribute breakdown:"]

    if attr_scores:
        max_name_len = max(len(name) for name in attr_scores)
        for attr_name, score in attr_scores.items():
            padded = attr_name.ljust(max_name_len)
            if score >= 1.0:
                annotation = "(exact match)"
            elif score > 0:
                penalty = score - 1.0
                annotation = f"[NEAR-MISS: penalty={penalty:+.2f}]"
            else:
                annotation = "(no match)"
            lines.append(f"    {padded}  {score:.2f}  {annotation}")
        avg_attr = sum(attr_scores.values()) / len(attr_scores)
    else:
        lines.append("    (no attribute filters)")
        avg_attr = 0.0

    lines.append(f"  FTS relevance:  {fts_score:.2f}")
    lines.append(f"  Sentiment:      {sent_score:.2f}")
    lines.append(
        f"  Weighted total: 0.50*{avg_attr:.2f} + 0.35*{fts_score:.2f} + 0.15*{sent_score:.2f} = {total_score:.3f}"
    )
    return "\n".join(lines)
