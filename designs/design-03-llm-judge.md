# Design #3: LLM-as-Judge / LLM-Centric Recommendation System

## 1. Architecture Overview

This design treats the local LLM as the primary reasoning engine at every stage of the pipeline. Rather than building elaborate scoring algorithms or feature-engineering attribute weights, we lean on the LLM's ability to understand nuance, context, and domain semantics directly.

The pipeline has two phases:

**Ingestion (offline):**
```
Raw Reviews --> [LLM Extraction] --> Structured JSON + Embeddings --> Vector Store
```

**Query (online):**
```
User Query --> [Embedding Recall] --> Top-K Candidates --> [LLM-as-Judge] --> Ranked Results + Explanations
```

The key architectural choice: embeddings handle **recall** (finding plausible candidates from thousands of products), but the LLM handles **precision** (evaluating which candidates actually match the user's intent, and why). This mirrors how RAG systems work, but instead of answering questions about documents, the LLM is judging product-preference fit.

Each LLM call is structured with a clear role: extractor, judge, or explainer. Prompts are domain-parameterized so the same system works for skis, shoes, cookies, or bikes without code changes.

## 2. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| LLM Runtime | Ollama / API providers | Local-first via Ollama; optional Anthropic or OpenAI API for faster/better models (see LLM Provider Abstraction below) |
| Extraction Model | `qwen2.5:7b` or `mistral:7b` | Good structured output, fast enough for batch ingestion |
| Judge Model | `qwen2.5:14b` or `llama3.1:8b` | Needs stronger reasoning for pointwise scoring |
| Embeddings | `nomic-embed-text` via Ollama | Solid retrieval performance, runs locally |
| Vector Store | ChromaDB | Embedded, zero-config, Python-native, good enough for POC scale |
| Orchestration | Python + `ollama` SDK | Direct Ollama API calls, no framework overhead |
| Structured Output | Pydantic + JSON mode | Enforce schema on LLM extraction output |
| Cache | SQLite | Cache LLM extractions and judgments to avoid redundant calls |

No LangChain. The orchestration logic is simple enough that a framework adds complexity without value. Direct LLM API calls with Pydantic validation keep things transparent and debuggable.

### LLM Provider Abstraction

The POC supports both local (Ollama) and remote (Anthropic, OpenAI) LLM providers via a simple abstraction. This lets benchmark runs swap in faster or higher-quality models without changing any pipeline code.

```python
class LLMProvider(Protocol):
    def chat(self, model: str, messages: list[dict], format: str = "") -> dict: ...
    def embed(self, model: str, input: str) -> dict: ...

class OllamaProvider:
    """Local inference via Ollama."""
    def chat(self, model, messages, format=""):
        return ollama.chat(model=model, messages=messages, format=format)
    def embed(self, model, input):
        return ollama.embed(model=model, input=input)

class AnthropicProvider:
    """Remote inference via Anthropic API."""
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    def chat(self, model, messages, format=""):
        resp = self.client.messages.create(model=model, messages=messages, max_tokens=2048)
        return {"message": {"content": resp.content[0].text}}
    def embed(self, model, input):
        raise NotImplementedError("Use Ollama or a dedicated embedding provider")

class OpenAIProvider:
    """Remote inference via OpenAI API."""
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    def chat(self, model, messages, format=""):
        resp = self.client.chat.completions.create(model=model, messages=messages)
        return {"message": {"content": resp.choices[0].message.content}}
    def embed(self, model, input):
        resp = self.client.embeddings.create(model=model, input=input)
        return {"embeddings": [resp.data[0].embedding]}
```

Configuration is via environment variable or constructor argument:

```python
# Default: local Ollama
provider = OllamaProvider()

# For benchmark runs with faster models:
provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
```

All pipeline code calls `provider.chat(...)` and `provider.embed(...)` instead of `ollama.chat(...)` directly.

## 3. Data Model

### Product Record (SQLite + ChromaDB)

```python
@dataclass
class ProductRecord:
    product_id: str              # stable identifier
    domain: str                  # "ski", "running_shoe", "cookie", etc.
    name: str                    # product name
    raw_reviews: list[str]       # original review texts
    extracted_profile: dict      # LLM-extracted structured attributes
    profile_summary: str         # LLM-generated natural language summary
    embedding: list[float]       # embedding of profile_summary
    source_urls: list[str]      # provenance
    extracted_at: datetime       # when extraction ran
```

### Extracted Profile (domain-flexible)

```json
{
  "domain": "ski",
  "name": "Nordica Enforcer 94",
  "attributes": {
    "flex": "stiff (8/10)",
    "terrain": ["on-piste", "hardpack", "light-crud"],
    "width_mm": 94,
    "turn_radius": "medium (16m at 177cm)",
    "weight": "moderate-heavy",
    "skill_level": "advanced to expert"
  },
  "consensus_summary": "Reviewers consistently praise the Enforcer 94 as a powerful on-piste carver that handles hardpack with confidence. The stiff flex and moderate weight make it less forgiving for intermediates. Several reviewers note it struggles in deep powder but excels in groomed conditions.",
  "reviewer_disagreements": "Mixed opinions on vibration dampening at speed.",
  "sentiment": "positive",
  "review_count": 12
}
```

The `attributes` dict is intentionally unschematized -- the LLM decides what matters per domain. A cookie might have `{"texture": "chewy", "sweetness": "moderate", "chocolate_ratio": "high"}`. This flexibility is a core advantage of the LLM-centric approach.

### Storage Layout

- **SQLite**: Stores product records, raw reviews, extracted profiles, and acts as a cache for LLM judgments.
- **ChromaDB**: Stores embeddings of `profile_summary` for vector retrieval. Metadata filters on `domain` field.

## 4. Ingestion Pipeline

### Step 1: Review Collection

Reviews arrive as unstructured text grouped by product. No assumptions about format.

### Step 2: LLM Extraction

Each product's reviews are batched into a single LLM call (or chunked if they exceed context). The extraction prompt is domain-parameterized:

```python
EXTRACTION_PROMPT = """You are analyzing {review_count} user reviews for a {domain} product: "{product_name}".

Extract a structured profile capturing what reviewers collectively say about this product.
Identify the key attributes that matter for {domain} products.
Note where reviewers agree and disagree.

Reviews:
{reviews_text}

Respond in JSON matching this structure:
{{
  "attributes": {{...key-value pairs of product characteristics...}},
  "consensus_summary": "...",
  "reviewer_disagreements": "...",
  "sentiment": "positive|mixed|negative",
  "review_count": {review_count}
}}"""
```

```python
async def extract_profile(product: RawProduct) -> ProductRecord:
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(
            domain=product.domain,
            product_name=product.name,
            review_count=len(product.reviews),
            reviews_text="\n---\n".join(product.reviews)
        )}],
        format="json"
    )
    profile = json.loads(response["message"]["content"])
    # Validate with Pydantic, store to SQLite
    # Generate embedding from consensus_summary, store to ChromaDB
    return ProductRecord(...)
```

### Step 3: Embedding Generation

The `consensus_summary` plus a flattened version of `attributes` gets embedded via `nomic-embed-text`. This embedding captures the "gist" of the product for retrieval. We embed the summary rather than raw reviews because it is already distilled and normalized by the extraction LLM.

### Throughput

At ~2-4 seconds per extraction call on a 7B model (Apple Silicon), a catalog of 500 products processes in ~20-30 minutes. This is a one-time cost per product, cached in SQLite.

## 5. Query Pipeline

### Stage 1: Embedding Recall (fast, broad)

```python
def recall_candidates(query: str, domain: str, top_k: int = 20) -> list[ProductRecord]:
    query_embedding = ollama.embed(model="nomic-embed-text", input=query)
    results = chroma_collection.query(
        query_embeddings=[query_embedding["embeddings"][0]],
        n_results=top_k,
        where={"domain": domain}
    )
    return load_products(results["ids"][0])
```

This retrieves 20 candidates in milliseconds. The embedding similarity acts as a coarse filter -- it will surface products in the right ballpark but cannot do precise preference matching.

### Stage 2: LLM-as-Judge (slow, precise)

The judge evaluates each candidate against the user's query using pointwise scoring.

**Pointwise scoring (all candidates):**

```python
JUDGE_PROMPT = """You are evaluating whether a {domain} product matches a user's preferences.

User's query: "{user_query}"

Product: {product_name}
Profile:
{product_profile_json}

Score this product's match to the user's query on a scale of 1-10.
Explain your reasoning in 2-3 sentences.
For each match_strength, also provide a numeric confidence from 0.0 to 1.0.

Respond in JSON:
{{"score": <1-10>, "reasoning": "...", "match_strengths": [{{"attribute": "...", "confidence": 0.0-1.0}}, ...], "match_gaps": [{{"attribute": "...", "severity": 0.0-1.0}}, ...]}}"""
```

**Score normalization:** Raw LLM scores (1-10) are normalized to 0-1 by dividing by 10. Because LLM scoring is inherently noisy, the POC runs each judgment with `temperature=0` and optionally averages 2-3 runs to stabilize scores for close-scoring products.

**Extracting `matched_attributes`:** The `match_strengths` and `match_gaps` from the judge output are parsed into a `matched_attributes` dict for the common interface. Each strength's `confidence` maps directly; each gap's `severity` is inverted (1.0 - severity) to represent how well the attribute matches. See Section 10 for the interface contract.

> **Note (future enhancement):** A pairwise tournament stage (comparing products head-to-head) could improve final ranking quality for the top-5 candidates. This is deferred from the POC to avoid doubling latency for marginal ranking improvement. Pointwise scoring alone is sufficient for benchmarking.

### Stage 3: Result Assembly

```python
async def query(user_query: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
    # Stage 1: Fast recall
    candidates = recall_candidates(user_query, domain, top_k=20)

    # Stage 2: Pointwise scoring (parallel, ~5 concurrent)
    # Check cache first, only call LLM for uncached (product_id, query_hash) pairs
    query_hash = hashlib.sha256(user_query.encode()).hexdigest()[:16]
    scored = await asyncio.gather(*[
        judge_pointwise_cached(user_query, query_hash, domain, c) for c in candidates
    ])

    # Normalize scores from 1-10 to 0-1
    for s in scored:
        s.score = s.score / 10.0

    ranked = sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]
    return ranked
```

## 6. Ranking Strategy

The ranking uses pointwise scoring to balance latency and quality.

**Pointwise scoring** runs all 20 candidates through the judge prompt concurrently. With Ollama handling 3-5 parallel requests on a decent GPU/Apple Silicon, this takes ~8-15 seconds for 20 candidates. Each product gets a 1-10 score (normalized to 0-1) with chain-of-thought reasoning.

To stabilize scores, all judge calls use `temperature=0`. For products with close scores (within 0.1 normalized), the system can optionally run 2-3 additional passes and average the results. This adds latency but produces more reliable final rankings.

> **Future enhancement:** A pairwise tournament tier taking the top-5 and running round-robin comparisons (10 pairs) could catch cases where the LLM's absolute scoring is inconsistent but its relative judgments are reliable. Deferred from POC scope.

**Scoring rubric injection:** The judge prompt can include domain-specific rubric hints:

```python
RUBRIC_HINTS = {
    "ski": "Consider: flex/stiffness match, terrain suitability, width appropriateness, skill level alignment",
    "running_shoe": "Consider: cushion level match, responsiveness, plate presence, drop height, intended pace",
    "cookie": "Consider: texture match, flavor profile, dietary restrictions, freshness indicators"
}
```

These are soft hints -- the LLM still reasons freely, but the rubric nudges it toward the attributes that matter most per domain.

**Chain-of-thought is implicit.** By asking for `reasoning`, `match_strengths`, and `match_gaps` in the JSON output, we force the LLM to reason before scoring. This consistently improves score quality vs. asking for a bare number.

## 7. Caching and Latency Mitigation

The 10-25 second query latency is the critical weakness of this design, especially for benchmarking where many queries run sequentially. Three concrete mitigations:

### 7a. Judge Result Cache

Cache every LLM judge result keyed by `(product_id, query_hash)` in SQLite:

```python
def get_cached_judgment(product_id: str, query_hash: str) -> dict | None:
    row = db.execute(
        "SELECT result_json FROM judge_cache WHERE product_id=? AND query_hash=?",
        (product_id, query_hash)
    ).fetchone()
    return json.loads(row[0]) if row else None

def cache_judgment(product_id: str, query_hash: str, result: dict):
    db.execute(
        "INSERT OR REPLACE INTO judge_cache (product_id, query_hash, result_json, cached_at) VALUES (?,?,?,?)",
        (product_id, query_hash, json.dumps(result), datetime.utcnow().isoformat())
    )
```

The `query_hash` is a SHA-256 prefix of the normalized query text. Identical queries hit cache immediately; semantically similar but textually different queries miss cache (acceptable for POC).

### 7b. Benchmark Cache Pre-warming

For the benchmark test query set, pre-warm the cache by running all (product, query) pairs ahead of time:

```python
def prewarm_cache(products: list[dict], test_queries: list[str], domain: str):
    """Run all judge evaluations upfront so benchmark timing reflects pure ranking logic."""
    for query in test_queries:
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        for product in products:
            if not get_cached_judgment(product["product_id"], query_hash):
                result = judge_pointwise(query, domain, product)
                cache_judgment(product["product_id"], query_hash, result)
```

This separates LLM inference time from benchmark measurement, letting us evaluate ranking quality without conflating it with latency.

### 7c. Batched Judging

Instead of one LLM call per product, batch multiple products into a single call:

```python
BATCH_JUDGE_PROMPT = """You are evaluating {n} {domain} products against a user's preferences.

User's query: "{user_query}"

{products_block}

For EACH product, score its match on 1-10 and explain. Respond in JSON:
{{"results": [{{"product_name": "...", "score": 1-10, "reasoning": "...", "match_strengths": [...], "match_gaps": [...]}}, ...]}}"""
```

Batching 5 products per call reduces 20 LLM calls to 4, cutting wall-clock time roughly in half (LLM calls are the bottleneck, not token count). Trade-off: slightly lower quality as the LLM juggles multiple evaluations. For the POC, batch size of 5 is a reasonable default.

## 8. Domain Adaptation

Adding a new domain requires zero code changes. The process:

1. **Provide reviews** for products in the new domain.
2. **Set the `domain` field** (e.g., `"bike"`, `"headphone"`).
3. **Optionally add a rubric hint** to the `RUBRIC_HINTS` dict.

The extraction LLM discovers relevant attributes from the reviews themselves. A cookie review mentioning "chewy center" and "crispy edges" will produce `{"texture": "chewy center with crispy edges"}` without anyone defining a texture taxonomy.

For higher quality in a specific domain, you can:
- Provide few-shot examples in the extraction prompt showing ideal attribute structures.
- Add domain-specific system prompts for the judge ("You are an expert ski technician...").
- Curate a rubric with weighted criteria.

But none of these are required. The baseline works across domains out of the box.

## 9. Pros and Cons

### Pros

- **Nuance handling.** The LLM understands "stiff but not punishing" or "cushy without feeling mushy" in ways that keyword matching or embedding similarity cannot. Semantic matching is the LLM's native capability.
- **Zero feature engineering.** No need to define attribute schemas, weight vectors, or scoring functions per domain. The LLM figures out what matters from context.
- **Explainability for free.** Every recommendation comes with natural language reasoning. Users see *why* a product was recommended, citing specific review evidence.
- **Domain agnostic by default.** New domains work immediately without schema definitions or training data.
- **Graceful handling of contradictory reviews.** The extraction step synthesizes disagreements, and the judge can reason about uncertainty.

### Cons

- **Latency.** The query pipeline takes 10-25 seconds with pointwise scoring. This is the biggest practical limitation. Mitigations: aggressive caching of judge results (Section 7a), batch judging (Section 7c), benchmark cache pre-warming (Section 7b), reducing candidate count, using faster models via the provider abstraction.
- **LLM inconsistency.** The same product-query pair may get different scores across runs. Mitigations: temperature=0, averaging 2-3 runs for close-scoring products.
- **Context window pressure.** Products with many reviews may exceed context limits during extraction. Mitigation: chunk reviews and merge extracted profiles.
- **Throughput ceiling.** Ollama on consumer hardware handles ~3-5 concurrent inference requests. This limits concurrent users to low single digits without queuing.
- **Harder to unit test.** LLM outputs are non-deterministic. Testing requires golden-set evaluation rather than exact-match assertions.
- **Cost of being wrong is opaque.** When the LLM misjudges a product-query match, it is harder to diagnose than a bug in a deterministic scoring function.

### Compared to embedding-only approaches

This design trades latency for ranking quality. An embedding-only system returns results in <1 second but cannot reason about complex multi-attribute queries ("stiff AND on-piste AND 180cm+ BUT not too heavy"). The LLM-as-judge can decompose, prioritize, and weigh these constraints.

## 10. POC Scope

**Goal:** End-to-end demo with 20-50 ski products, showing query-to-ranked-results with explanations.

**Components to build:**

```
poc/
  ingest.py        # Read reviews, call LLM extraction, store to SQLite + Chroma
  query.py         # Query interface: recall + judge + display results
  models.py        # Pydantic models for ProductRecord, JudgeResult, RecommendationResult, etc.
  prompts.py       # All prompt templates
  db.py            # SQLite and ChromaDB helpers
  providers.py     # LLM provider abstraction (Ollama, Anthropic, OpenAI)
  cache.py         # Judge result caching and pre-warming
  interface.py     # Common Recommender interface implementation
  sample_data/     # 20-50 ski products with 5-10 reviews each
```

**Minimal ingest.py sketch:**

```python
import json, sqlite3, chromadb
from models import ProductRecord
from prompts import EXTRACTION_PROMPT
from providers import LLMProvider

def ingest_product(name: str, domain: str, reviews: list[str], db, chroma, provider: LLMProvider):
    # Extract structured profile
    resp = provider.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(
            domain=domain, product_name=name,
            review_count=len(reviews),
            reviews_text="\n---\n".join(reviews)
        )}],
        format="json"
    )
    profile = json.loads(resp["message"]["content"])

    # Generate embedding from summary
    summary_text = f"{name}: {profile['consensus_summary']}"
    emb = provider.embed(model="nomic-embed-text", input=summary_text)

    # Store
    db.execute(
        "INSERT INTO products (name, domain, profile_json, raw_reviews) VALUES (?,?,?,?)",
        (name, domain, json.dumps(profile), json.dumps(reviews))
    )
    chroma.add(ids=[name], embeddings=[emb["embeddings"][0]],
               metadatas=[{"domain": domain}], documents=[summary_text])
```

**Minimal query.py sketch:**

```python
import asyncio, json, hashlib
from prompts import JUDGE_PROMPT
from providers import LLMProvider
from cache import get_cached_judgment, cache_judgment
from interface import RecommendationResult, build_matched_attributes

async def judge_one(query: str, domain: str, product: dict, provider: LLMProvider) -> dict:
    resp = provider.chat(
        model="qwen2.5:14b",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            domain=domain, user_query=query,
            product_name=product["name"],
            product_profile_json=json.dumps(product["profile"], indent=2)
        )}],
        format="json"
    )
    return json.loads(resp["message"]["content"])

async def recommend(query: str, domain: str = "ski", provider: LLMProvider = None, top_k: int = 10):
    # Recall
    emb = provider.embed(model="nomic-embed-text", input=query)
    candidates = chroma.query(query_embeddings=[emb["embeddings"][0]],
                              n_results=20, where={"domain": domain})

    # Judge (with caching)
    products = load_from_sqlite(candidates["ids"][0])
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    results = []
    for p in products:
        cached = get_cached_judgment(p["product_id"], query_hash)
        if cached:
            results.append(cached)
        else:
            result = await judge_one(query, domain, p, provider)
            cache_judgment(p["product_id"], query_hash, result)
            results.append(result)

    # Rank, normalize, and display
    ranked = sorted(zip(products, results), key=lambda x: x[1]["score"], reverse=True)
    for product, result in ranked[:top_k]:
        score_normalized = result["score"] / 10.0
        print(f"\n{'='*60}")
        print(f"{product['name']} -- Score: {score_normalized:.2f}")
        print(f"Why: {result['reasoning']}")
        attrs = build_matched_attributes(result)
        print(f"Matched attributes: {attrs}")

# Usage: asyncio.run(recommend("stiff, on-piste carving ski, 180cm+", provider=OllamaProvider()))
```

**POC success criteria:**
- Ingestion processes 20+ products with valid JSON extraction (>90% parse success rate).
- Query returns top-10 ranked results in <30 seconds on Apple Silicon (uncached), <2 seconds (cached).
- Results for "stiff, on-piste carving ski" demonstrably differ from "playful, all-mountain ski" (the ranking should change, not just the explanations).
- Explanations reference actual review content, not hallucinated features.
- Output conforms to the `Recommender` protocol (Section 11) with normalized 0-1 scores and populated `matched_attributes`.

**What the POC deliberately skips:** authentication, web UI, concurrent users, pairwise tournament (see Section 6 note), multi-domain testing. These are all straightforward to add once the core LLM extraction and judging prompts are validated.

## 11. Common Interface

All recommender designs implement a shared interface so the benchmark harness can evaluate them interchangeably.

### Interface Definition

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

### Score Normalization

The LLM judge returns scores on a 1-10 scale. These are normalized to 0-1 by dividing by 10. For example, a judge score of 8 becomes 0.8. This makes scores comparable across different recommender implementations that may use different internal scales.

### Mapping `matched_attributes`

The judge prompt returns `match_strengths` and `match_gaps` with numeric values. These are mapped to the `matched_attributes` dict as follows:

```python
def build_matched_attributes(judge_result: dict) -> dict[str, float]:
    attrs = {}
    for strength in judge_result.get("match_strengths", []):
        # Each strength has {"attribute": "...", "confidence": 0.0-1.0}
        attrs[strength["attribute"]] = strength["confidence"]
    for gap in judge_result.get("match_gaps", []):
        # Each gap has {"attribute": "...", "severity": 0.0-1.0}
        # Invert severity: a severe gap (1.0) means low match (0.0)
        attrs[gap["attribute"]] = 1.0 - gap["severity"]
    return attrs
```

### Implementation Sketch

```python
class LLMJudgeRecommender:
    """Implements the Recommender protocol using the LLM-as-Judge pipeline."""

    def __init__(self, provider: LLMProvider, judge_model: str = "qwen2.5:14b"):
        self.provider = provider
        self.judge_model = judge_model
        self.db = None
        self.chroma = None

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        # Run extraction pipeline (Section 4)
        for product in products:
            product_reviews = [r for r in reviews if r["product_id"] == product["id"]]
            extract_and_store(product, product_reviews, domain, self.provider)

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        # Recall via embeddings
        candidates = recall_candidates(query_text, domain, top_k=20)

        # Judge with caching
        query_hash = hashlib.sha256(query_text.encode()).hexdigest()[:16]
        results = []
        for c in candidates:
            cached = get_cached_judgment(c["product_id"], query_hash)
            if cached:
                judge_result = cached
            else:
                judge_result = judge_pointwise(query_text, domain, c, self.provider)
                cache_judgment(c["product_id"], query_hash, judge_result)

            results.append(RecommendationResult(
                product_id=c["product_id"],
                product_name=c["name"],
                score=judge_result["score"] / 10.0,
                explanation=judge_result["reasoning"],
                matched_attributes=build_matched_attributes(judge_result),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
```
