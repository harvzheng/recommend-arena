# Design #6: Multi-Agent Architecture

A recommendation system built around specialized, collaborating agents orchestrated via LangGraph. Each agent owns a distinct responsibility -- extraction, query understanding, retrieval, ranking, and explanation -- communicating through a shared state graph rather than ad-hoc function calls.

---

## 1. Architecture Overview

Five agents form a directed acyclic graph (DAG) managed by LangGraph:

```
                          INGESTION PATH
                          ==============
    Raw Reviews --> [Extractor Agent] --> Structured Store + Vector Store

                          QUERY PATH
                          ==========
    User Query --> [Query Understanding Agent]
                         |
                         v
                  [Retrieval Agent]
                         |
                         v
                  [Ranking Agent]
                         |
                         v
                  [Explanation Agent] --> Final Response
```

**Agent roles:**

| Agent | Responsibility | LLM-powered? |
|---|---|---|
| **Extractor** | Parse reviews into structured attributes + sentiment per attribute | Yes |
| **Query Understanding** | Normalize user intent into a structured query spec (filters, soft prefs, domain) | Yes |
| **Retrieval** | Fetch candidate products via vector similarity + hard filters | No (deterministic) |
| **Ranking** | Score and order candidates against the parsed query | Hybrid (heuristic + optional LLM rerank) |
| **Explanation** | Generate human-readable justifications for the top-N results | Yes |

**Communication pattern:** Agents do not call each other directly. They read from and write to a shared `RecommendationState` TypedDict that flows through the LangGraph graph. Each agent node receives the full state, performs its work, and returns a state patch. This makes the system fully inspectable at every step.

**Orchestration:** LangGraph's `StateGraph` handles sequencing, conditional edges (e.g., skip Explanation if the caller requests raw scores), and retry/fallback logic. No autonomous agent loops -- every path through the graph is predetermined and bounded.

---

## 2. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Agent framework | **LangGraph** (v0.2+) | Explicit graph over autonomous agents; debuggable, no runaway loops |
| LLM runtime | **Ollama** (local) | Runs Mistral/Llama 3 locally; structured output via grammar mode |
| Vector store | **ChromaDB** | Embedded, zero-config, persistent, good Python API |
| Structured storage | **SQLite** | Zero-config, single-file, consistent with other designs; sufficient for POC-scale attribute queries |
| Embedding model | **nomic-embed-text** via Ollama | Strong quality for its size; runs locally |
| Schema validation | **Pydantic v2** | Validates every agent's input/output contract |
| Serialization | **JSON** for state, **Parquet** for bulk attribute storage |

All components run locally. No API keys, no cloud dependencies.

---

## 3. Data Model

### Shared Agent State

```python
from typing import TypedDict, Optional
from pydantic import BaseModel

class ProductAttributes(BaseModel):
    product_id: str
    product_name: str
    domain: str                          # "ski", "running_shoe", "cookie", ...
    attributes: dict[str, float | str]   # e.g. {"stiffness": 0.85, "terrain": "on-piste"}
    sentiment_scores: dict[str, float]   # per-attribute sentiment from reviews
    review_count: int
    avg_rating: float

class ParsedQuery(BaseModel):
    domain: str
    hard_filters: dict[str, str | float] # must-match constraints
    soft_preferences: dict[str, float]   # weighted preferences (-1 to 1 scale)
    freetext_intent: str                 # original query preserved

class ScoredCandidate(BaseModel):
    product: ProductAttributes
    score: float
    breakdown: dict[str, float]          # per-attribute contribution to score
    explanation: str = ""

class RecommendationState(TypedDict):
    # Set by Query Understanding Agent
    raw_query: str
    parsed_query: Optional[ParsedQuery]
    # Set by Retrieval Agent
    candidates: list[ProductAttributes]
    # Set by Ranking Agent
    ranked: list[ScoredCandidate]
    # Set by Explanation Agent
    response: str
    # Metadata
    domain: str
    errors: list[str]
```

### Persistent Storage

**SQLite tables:**
- `products` -- one row per product with flattened top-level fields
- `product_attributes` -- EAV table: `(product_id, attr_name, attr_value_num, attr_value_str)` for flexible per-domain schemas
- `reviews_raw` -- original review text with `product_id` foreign key

**ChromaDB collections:**
- `product_summaries` -- one embedding per product (concatenated attribute summary)
- `review_chunks` -- individual review segments for fine-grained retrieval

---

## 4. Ingestion Pipeline (Extractor Agent)

The Extractor Agent runs offline over batches of reviews. It is the only agent that writes to persistent storage.

```
Raw reviews (grouped by product)
    |
    v
[1] Domain detection (LLM classifies domain if not provided)
    |
    v
[2] Batch attribute extraction (LLM extracts attributes from ALL reviews for a product in one call)
    |
    v
[3] Cross-review aggregation (deterministic: normalize and merge extracted attributes)
    |
    v
[4] Persist to SQLite + embed summaries into ChromaDB
```

**Step 2 detail -- batched extraction prompt pattern:**

Reviews are grouped by `product_id` before extraction. All reviews for a single product are sent in one LLM call. This is more efficient (fewer calls) and produces better results because the LLM can synthesize across reviews -- e.g., resolving contradictions ("one reviewer says stiff, another says medium") and weighting by review quality.

```python
EXTRACT_PROMPT = """You are analyzing {review_count} reviews for a {domain} product: "{product_name}".

Reviews:
{numbered_reviews}

Extract structured attributes by synthesizing across ALL reviews.
Where reviewers disagree, weight toward the majority or more detailed reviews.
Return JSON matching this schema:
{schema_json}

Only extract attributes with sufficient evidence. Use null for unmentioned attributes.
Include a confidence score (0-1) for each attribute based on reviewer agreement."""
```

The schema is loaded from a domain config file (see Section 7). Ollama's structured output mode (JSON grammar) enforces valid output without brittle parsing.

**Aggregation logic:** For numeric attributes, take the weighted mean across reviews (weighted by review helpfulness or recency). For categorical attributes, take the mode. Sentiment scores are averaged per attribute. Because extraction is now batched per product, much of this aggregation happens within the LLM call itself, with the deterministic step handling normalization and any cross-batch merging.

---

## 5. Query Pipeline (Agent Collaboration)

The query path is a linear LangGraph `StateGraph` with a conditional edge to skip explanation when not needed:

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(RecommendationState)

graph.add_node("understand", query_understanding_agent)
graph.add_node("retrieve", retrieval_agent)
graph.add_node("rank", ranking_agent)
graph.add_node("explain", explanation_agent)

graph.set_entry_point("understand")
graph.add_edge("understand", "retrieve")
graph.add_edge("retrieve", "rank")

# Explanation is optional -- skip it for benchmark/fast mode
def should_explain(state):
    return "explain" if state.get("explain_results", True) else END

graph.add_conditional_edges("rank", should_explain)
graph.add_edge("explain", END)

app = graph.compile()
```

**Query Understanding Agent** takes the raw query string and produces a `ParsedQuery`. This is the critical LLM step -- it must correctly identify the domain, separate hard filters (size = 180cm) from soft preferences (stiff, on-piste leaning), and normalize attribute names to match the domain schema.

The agent includes retry and fallback logic for LLM parse failures (see Section 10 for details):

```python
def query_understanding_agent(state: RecommendationState) -> dict:
    max_retries = 2
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(
                QUERY_PARSE_PROMPT.format(
                    query=state["raw_query"],
                    domain_schemas=load_all_domain_schemas()
                ),
                response_format=ParsedQuery
            )
            parsed = ParsedQuery.model_validate_json(response)
            return {"parsed_query": parsed, "domain": parsed.domain}
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            if attempt < max_retries:
                continue  # Retry -- Ollama JSON mode occasionally produces trailing text

    # Fallback: treat entire query as freetext, use vector search only
    fallback = ParsedQuery(
        domain=state.get("domain", "unknown"),
        hard_filters={},
        soft_preferences={},
        freetext_intent=state["raw_query"]
    )
    return {
        "parsed_query": fallback,
        "domain": fallback.domain,
        "errors": state.get("errors", []) + [f"Query parse failed after {max_retries + 1} attempts: {last_error}. Using freetext fallback."]
    }
```

**Retrieval Agent** is deterministic. It applies hard filters via SQLite SQL, then fetches the top-K similar products from ChromaDB using the soft preferences as the query embedding. Union of both result sets forms the candidate pool (typically 20-50 products).

**Ranking Agent** scores each candidate (see Section 6). It produces a per-attribute score breakdown in `ScoredCandidate.breakdown`, which is sufficient to populate the common interface's `explanation` field without an LLM call.

**Explanation Agent** (optional) takes the top-N ranked candidates and generates a natural language response citing specific review evidence. It retrieves relevant review chunks from ChromaDB for each recommended product. For benchmark runs, this agent is skipped -- the `explanation` field in `RecommendationResult` is built from the Ranking Agent's score breakdown instead (e.g., `"Matched stiffness (0.92), terrain (1.0), sentiment positive"`). This reduces query-path LLM calls from three to one.

---

## 6. Ranking Strategy

The Ranking Agent uses a two-phase approach:

### Phase 1: Heuristic Scoring (always runs)

For each candidate, compute a weighted attribute-match score:

```python
def score_candidate(product: ProductAttributes, query: ParsedQuery) -> float:
    score = 0.0
    total_weight = 0.0

    for attr, preference in query.soft_preferences.items():
        if attr in product.attributes:
            product_val = product.attributes[attr]
            if isinstance(product_val, (int, float)):
                # Cosine-like alignment: preference is -1..1, product_val is 0..1
                match = 1.0 - abs(preference - product_val)
            else:
                match = 1.0 if product_val == preference else 0.0

            # Weight by review confidence (more reviews = more trust)
            confidence = min(product.review_count / 10, 1.0)
            sentiment_boost = product.sentiment_scores.get(attr, 0.0)

            attr_score = match * confidence * (1 + 0.2 * sentiment_boost)
            score += attr_score
            total_weight += 1.0

    return score / max(total_weight, 1.0)
```

### Phase 2: LLM Rerank (optional, top-10 only)

For higher-quality results at the cost of latency, the top 10 candidates from Phase 1 are passed to the LLM for pairwise or listwise reranking. This catches nuances the heuristic misses (e.g., "playful but not noodly" requires understanding attribute relationships).

The rerank step is behind a feature flag. For the POC, Phase 1 alone is sufficient.

---

## 7. Domain Adaptation

Each domain is defined by a YAML config file:

```yaml
# domains/ski.yaml
domain: ski
display_name: "Skis"
attributes:
  stiffness:
    type: numeric        # 0.0 to 1.0 normalized
    description: "Flex pattern from soft/forgiving to stiff/demanding"
    synonyms: ["flex", "rigidity", "stiff", "soft"]
  terrain:
    type: categorical
    values: ["on-piste", "off-piste", "all-mountain", "park"]
    description: "Primary intended terrain"
    synonyms: ["groomer", "powder", "backcountry"]
  length_cm:
    type: numeric_raw    # not normalized, actual value
    description: "Ski length in centimeters"
  turn_radius:
    type: numeric_raw
    description: "Turn radius in meters"
```

**How agents use domain configs:**

- **Extractor Agent**: The attribute list and descriptions are injected into the extraction prompt. The LLM knows exactly which attributes to look for and how to normalize them.
- **Query Understanding Agent**: Synonym lists help map user language ("groomer ski") to canonical attribute values ("terrain: on-piste"). The schema is included in the query-parsing prompt.
- **Ranking Agent**: Attribute types determine the scoring function (numeric similarity vs. categorical match).

Adding a new domain means writing one YAML file. No code changes, no retraining. Agents dynamically load the config for the detected domain.

---

## 8. Justifying the Architecture: Why Multi-Agent?

The honest answer: for the POC, a linear pipeline of plain functions would produce identical results. The query path is a fixed sequence (understand -> retrieve -> rank -> optionally explain), and there are no dynamic routing decisions that require a graph runtime.

**What LangGraph adds beyond plain function calls:**

1. **State checkpointing.** LangGraph persists the full `RecommendationState` at each node boundary. This means any query can be replayed from any intermediate step without re-running earlier stages -- useful for debugging bad recommendations ("was the parse wrong, or the ranking?").
2. **Conditional edges without if/else spaghetti.** The optional explanation step, the optional rerank step, and future additions (e.g., a safety-check agent, a cache-hit shortcut) are expressed as graph topology rather than nested conditionals in a main function.
3. **Visual tracing.** LangGraph Studio renders the execution graph with timing and state at each node. For a system with LLM calls that can fail in subtle ways, this is more valuable than log statements.

**What it does not add:** autonomy, loops, or dynamic tool selection. This is not an "agentic" system in the AutoGPT sense. It is a structured pipeline with the option to grow into something more dynamic later.

**Design decision:** The implementation provides a `use_langgraph: bool` toggle. When `False`, the same agent functions are called in sequence by a plain Python orchestrator. When `True`, LangGraph manages the flow. The benchmark defaults to `False` for simplicity; the LangGraph path exists to demonstrate the architecture and can be enabled for debugging or future extensions. This way the framework serves the design, not the other way around.

---

## 9. Pros and Cons

### Strengths

- **Separation of concerns.** Each agent can be developed, tested, and improved independently. Swap the ranking algorithm without touching extraction.
- **Inspectability.** The shared state is a complete audit trail. Every intermediate result (parsed query, candidate list, scores, breakdowns) is available for debugging. LangGraph Studio provides visual tracing when the graph orchestrator is enabled.
- **Flexible quality/latency tradeoff.** LLM-powered steps (reranking, explanation) can be toggled off for faster responses. In benchmark mode, the query path uses only one LLM call (query understanding), with fallback to pure vector similarity if that call fails.
- **Domain agnostic by design.** YAML-driven attribute schemas mean new domains require zero code changes.
- **Graceful degradation.** If the LLM produces malformed JSON, the Query Understanding Agent retries (up to 2 times) then falls back to freetext vector search. Each agent boundary is a natural error-handling point.
- **Framework-optional.** The plain-function orchestrator means LangGraph is a dependency you opt into, not one you are locked into.

### Weaknesses

- **Orchestration overhead.** Even with the toggle, maintaining two orchestration paths (plain functions and LangGraph) adds surface area. The LangGraph path is justified primarily by debugging affordances, not runtime behavior.
- **Latency.** Even with explanation disabled, the query understanding LLM call at local Ollama speeds adds 2-5 seconds per query. Enabling all optional LLM steps brings this to 5-15 seconds.
- **State bloat.** Passing full candidate lists through the state graph means serializing/deserializing potentially large objects at each step. Manageable for dozens of products; problematic at scale.
- **Testing complexity.** Integration tests must mock LLM responses at multiple agent boundaries. Unit testing individual agents is straightforward; end-to-end tests are harder.
- **Debugging distributed logic.** When recommendations are bad, the cause could be in any of five agents. The audit trail helps, but root-cause analysis still requires checking each stage.

---

## 10. Common Interface

All designs share a common interface so the benchmark harness can drive them uniformly. This design's multi-agent internals are hidden behind the `Recommender` protocol:

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

**Implementation sketch:**

```python
class MultiAgentRecommender:
    """Wraps the multi-agent pipeline behind the common Recommender interface."""

    def __init__(self, use_langgraph: bool = False, explain: bool = False):
        self.use_langgraph = use_langgraph
        self.explain = explain
        self.db = sqlite3.connect("products.db")
        self.chroma = chromadb.PersistentClient(path="./chroma_store")

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Run the Extractor Agent over batched reviews, persist to SQLite + ChromaDB."""
        reviews_by_product = group_reviews_by_product(reviews)
        for product_id, product_reviews in reviews_by_product.items():
            attrs = extract_attributes_batch(product_reviews, domain)
            store_to_sqlite(self.db, product_id, attrs)
            embed_to_chroma(self.chroma, product_id, attrs)

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        """Run the query pipeline and return normalized results."""
        state = {
            "raw_query": query_text,
            "domain": domain,
            "explain_results": self.explain,
            "errors": [],
        }

        if self.use_langgraph:
            result = self.graph.invoke(state)
        else:
            state = understand(state)
            state = retrieve(state)
            state = rank(state)
            if self.explain:
                state = explain(state)

        # Convert internal ScoredCandidates to common interface
        return [
            RecommendationResult(
                product_id=c.product.product_id,
                product_name=c.product.product_name,
                score=c.score,
                explanation=c.explanation or _build_explanation_from_breakdown(c.breakdown),
                matched_attributes=c.breakdown,
            )
            for c in state["ranked"][:top_k]
        ]


def _build_explanation_from_breakdown(breakdown: dict[str, float]) -> str:
    """Build a human-readable explanation from the score breakdown without an LLM call."""
    parts = [f"{attr} ({score:.2f})" for attr, score in sorted(breakdown.items(), key=lambda x: -x[1])]
    return f"Matched: {', '.join(parts)}" if parts else "General match based on review similarity."
```

The benchmark harness calls `ingest()` once per domain, then `query()` for each test query. It never touches LangGraph, agent state, or internal data structures.

---

## 11. POC Scope

**Goal:** Demonstrate the full query path with 20 pre-extracted ski products. Ingestion can be a script rather than a full agent workflow.

### Minimal Demo Components

1. **Pre-seeded data**: 20 skis with hand-curated attributes in SQLite + ChromaDB embeddings
2. **Query Understanding Agent**: Ollama + structured output, ski domain only, with retry + fallback
3. **Retrieval Agent**: ChromaDB similarity + SQLite filter, no LLM
4. **Ranking Agent**: Heuristic scoring only (no rerank), builds explanation from score breakdown
5. **Common Interface**: `MultiAgentRecommender` wrapping the pipeline, `explain=False` by default
6. **Explanation Agent**: Disabled for benchmark; available via `explain=True` flag

### Code Sketch: End-to-End Query

```python
# poc.py
import json
import sqlite3
import chromadb
from pydantic import BaseModel, ValidationError
from ollama import chat

# -- Domain config (inline for POC) --
SKI_SCHEMA = {
    "stiffness": "numeric",
    "terrain": "categorical",
    "length_cm": "numeric_raw",
}

# -- Agents --
def understand(state):
    max_retries = 2
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = chat(
                model="mistral",
                messages=[{
                    "role": "user",
                    "content": f"Parse this ski query into structured filters and preferences.\n"
                               f"Query: {state['raw_query']}\n"
                               f"Available attributes: {list(SKI_SCHEMA.keys())}\n"
                               f"Return JSON with hard_filters and soft_preferences."
                }],
                format="json"
            )
            parsed = json.loads(response.message.content)
            result = ParsedQuery(domain="ski", **parsed)
            return {**state, "parsed_query": result}
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            if attempt < max_retries:
                continue
    # Fallback: freetext-only query
    fallback = ParsedQuery(
        domain="ski", hard_filters={}, soft_preferences={},
        freetext_intent=state["raw_query"]
    )
    return {
        **state,
        "parsed_query": fallback,
        "errors": state.get("errors", []) + [f"Parse failed: {last_error}"]
    }

def retrieve(state):
    q = state["parsed_query"]
    db = sqlite3.connect("products.db")

    # Hard filters via SQL (parameterized)
    where_clauses = []
    params = []
    for attr, val in q.hard_filters.items():
        where_clauses.append(f"{attr} = ?")
        params.append(val)
    where = " AND ".join(where_clauses) if where_clauses else "1=1"
    sql_results = db.execute(f"SELECT * FROM products WHERE {where}", params).fetchall()

    # Soft preferences via vector similarity
    chroma = chromadb.PersistentClient(path="./chroma_store")
    collection = chroma.get_collection("ski_products")
    query_text = " ".join(f"{k}:{v}" for k, v in q.soft_preferences.items())
    vector_results = collection.query(query_texts=[query_text], n_results=20)

    candidates = merge_and_dedupe(sql_results, vector_results)
    return {**state, "candidates": candidates}

def rank(state):
    scored = []
    for product in state["candidates"]:
        s, breakdown = score_candidate_with_breakdown(product, state["parsed_query"])
        explanation = _build_explanation_from_breakdown(breakdown)
        scored.append(ScoredCandidate(
            product=product, score=s, breakdown=breakdown, explanation=explanation
        ))
    scored.sort(key=lambda x: x.score, reverse=True)
    return {**state, "ranked": scored[:5]}

def explain(state):
    """Optional: enrich top results with LLM-generated explanations."""
    top = state["ranked"][:3]
    product_summaries = "\n".join(
        f"- {c.product.product_name} (score: {c.score:.2f})" for c in top
    )
    response = chat(
        model="mistral",
        messages=[{
            "role": "user",
            "content": f"Original query: {state['raw_query']}\n"
                       f"Top recommendations:\n{product_summaries}\n"
                       f"Write a brief, helpful explanation of why these skis match."
        }]
    )
    return {**state, "response": response.message.content}

# -- Plain function orchestrator (default for benchmark) --
def run_query(raw_query: str, explain_results: bool = False) -> dict:
    state = {"raw_query": raw_query, "errors": []}
    state = understand(state)
    state = retrieve(state)
    state = rank(state)
    if explain_results:
        state = explain(state)
    return state

# -- Run --
result = run_query("stiff, on-piste carving ski, 180cm+")
for c in result["ranked"]:
    print(f"{c.product.product_name}: {c.score:.2f} -- {c.explanation}")
```

### POC Milestones

1. Seed SQLite + ChromaDB with 20 ski products (day 1)
2. Wire plain-function orchestrator with understand/retrieve/rank agents (day 2)
3. Implement `MultiAgentRecommender` common interface, run benchmark (day 3)
4. Add LangGraph wrapper with conditional explain edge, verify parity (day 3, stretch)
5. Add a second domain (running shoes) via YAML config only (day 4, stretch)
