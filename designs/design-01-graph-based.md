# Design #1: Graph-Based Knowledge Graph

## 1. Architecture Overview

The system models the entire recommendation domain as a **property graph** where products, attributes, attribute values, and reviews are first-class nodes connected by typed, weighted edges. Sentiment extracted from reviews becomes edge weight data rather than a separate data store.

```
[Review] --MENTIONS{sentiment: 0.9}--> [Attribute:Stiffness]
[Product] --HAS_ATTRIBUTE{value: "stiff", confidence: 0.85}--> [Attribute:Stiffness]
[Product] --BELONGS_TO--> [Category:Carving Ski]
[Attribute:Stiffness] --SIMILAR_TO{weight: 0.7}--> [Attribute:Flex Pattern]
```

The key architectural idea: a user query like "stiff, on-piste carving ski" is parsed into a **query subgraph** — a small pattern of attribute nodes with desired polarities. Recommendation becomes **subgraph matching**: find product nodes whose neighborhoods best overlap the query subgraph.

This avoids embedding-space retrieval entirely. The graph is the index.

## 2. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Graph store | **NetworkX** (in-memory) | No server dependency, fast for <10K nodes, pure Python. Sufficient for benchmark scale (50-200 products). |
| Persistence | **SQLite** + pickled graph | SQLite for raw reviews/products, pickle or GraphML for the graph snapshot. |
| ABSA extraction | **LLM-based** via Ollama (llama3 8B) or API | Structured extraction of (aspect, opinion, sentiment) triples. No rule-based fallback — LLM handles cold start natively. |
| Query parsing | **spaCy** (en_core_web_sm) + custom attribute matcher | Extracts attribute mentions, polarities, and hard constraints from natural language. |
| LLM fallback | **Ollama** (llama3 8B or phi3) | Optional. Used only for ambiguous query interpretation or domain bootstrapping. Local. |
| API layer | **FastAPI** | Lightweight, async-ready, easy to prototype. |
| Frontend | None (CLI + JSON API for POC) | |

Total RAM footprint for the benchmark scale (200 products, 2000 reviews): under 50 MB. Fits trivially on consumer hardware.

## 3. Data Model

### Node Types

```python
# Product node
{
    "type": "product",
    "id": "ski-001",
    "name": "Volkl Deacon 80",
    "domain": "ski",
    "meta": {"length_cm": [165, 172, 180], "year": 2025, "price": 799}
}

# Attribute node (domain-level, shared across products)
{
    "type": "attribute",
    "id": "attr-stiffness",
    "name": "stiffness",
    "domain": "ski",
    "spectrum": ["soft", "medium", "stiff"]  # ordered value scale
}

# Review node
{
    "type": "review",
    "id": "rev-00421",
    "product_id": "ski-001",
    "text": "Incredibly stiff and damp. Hooks up on ice like nothing else.",
    "source": "blister_review",
    "date": "2025-11-03"
}

# Category node
{
    "type": "category",
    "id": "cat-carving",
    "name": "carving",
    "domain": "ski"
}
```

### Edge Types

| Edge | From | To | Properties |
|---|---|---|---|
| `HAS_ATTRIBUTE` | Product | Attribute | `value: str`, `score: float [-1,1]`, `confidence: float`, `mention_count: int` |
| `REVIEWED_BY` | Product | Review | `date`, `source` |
| `MENTIONS` | Review | Attribute | `sentiment: float [-1,1]`, `snippet: str` |
| `BELONGS_TO` | Product | Category | `primary: bool` |
| `SIMILAR_TO` | Attribute | Attribute | `weight: float` (semantic similarity) |
| `COMPATIBLE_WITH` | Product | Product | `reason: str` (optional, for bundles) |

The `HAS_ATTRIBUTE` edge is the **aggregated** signal — computed from all `MENTIONS` edges for that product-attribute pair. It is the primary edge used during query resolution.

## 4. Ingestion Pipeline

```
Raw Reviews (JSON/CSV)
    │
    ▼
┌─────────────────────┐
│  1. Text Cleaning    │  Strip HTML, normalize unicode, segment sentences
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  2. ABSA Extraction  │  For each sentence → (aspect_term, opinion_term, sentiment)
└─────────┬───────────┘  e.g. ("stiffness", "incredibly stiff", +0.92)
          ▼
┌─────────────────────┐
│  3. Attribute        │  Map extracted aspect terms to canonical attribute nodes
│     Normalization    │  "flex" / "stiffness" / "rigidity" → attr-stiffness
└─────────┬───────────┘  Uses synonym dict + embedding similarity fallback
          ▼
┌─────────────────────┐
│  4. Graph Upsert     │  Create/update nodes and edges
└─────────┬───────────┘  Aggregate MENTIONS into HAS_ATTRIBUTE scores
          ▼
┌─────────────────────┐
│  5. Snapshot         │  Persist graph to disk (pickle/GraphML)
└─────────────────────┘
```

### Aggregation logic for `HAS_ATTRIBUTE`

```python
def aggregate_attribute(product_id: str, attr_id: str, graph: nx.DiGraph) -> dict:
    """Compute aggregate score from all review mentions."""
    mentions = [
        graph.edges[rev, attr_id]
        for rev in graph.predecessors(product_id)  # reviews of this product
        if graph.has_edge(rev, attr_id)
    ]
    if not mentions:
        return None
    sentiments = [m["sentiment"] for m in mentions]
    return {
        "score": sum(sentiments) / len(sentiments),  # mean sentiment
        "confidence": min(1.0, len(sentiments) / 5),  # saturates at 5 mentions
        "mention_count": len(sentiments),
    }
```

## 5. Query Pipeline

```
User query: "stiff, damp carving ski, 180cm+"
    │
    ▼
┌──────────────────────┐
│  1. Parse Query       │  spaCy NLP + attribute matcher
│                       │  → attributes: [{stiffness: +1}, {dampness: +1}]
│                       │  → categories: [carving]
│                       │  → constraints: [length_cm >= 180]
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  2. Build Query       │  Construct a virtual "ideal product" subgraph:
│     Subgraph          │  node=query_product → edges to attr nodes with target scores
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  3. Candidate Filter  │  Category filter: only products in "carving"
│                       │  Constraint filter: length_cm >= 180
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  4. Graph Match &     │  Score each candidate against query subgraph
│     Rank              │  (see Ranking Strategy below)
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  5. Explain           │  For top-K results, pull supporting snippets
│                       │  from MENTIONS edges for justification
└──────────────────────┘
```

### Query attribute matching

The attribute matcher is a two-stage lookup:

1. **Exact match** against the domain's attribute synonym dictionary.
2. **Fuzzy match** using spaCy word vectors (cosine similarity > 0.75 against known attribute names).

Polarity is inferred from modifier words ("stiff" → positive on stiffness, "not too stiff" → negative) using dependency parsing.

## 6. Ranking Strategy

Each candidate product is scored against the query subgraph using a weighted sum:

```python
def score_product(product_id: str, query_attrs: list[dict], graph: nx.DiGraph) -> float:
    total, weight_sum = 0.0, 0.0
    for qa in query_attrs:
        attr_id = qa["attr_id"]
        desired = qa["polarity"]       # +1 or -1
        importance = qa["weight"]      # default 1.0, boosted for explicit emphasis
        edge = graph.edges.get((product_id, attr_id))
        if edge is None:
            continue  # no data — neutral, not penalized
        alignment = edge["score"] * desired  # [-1, +1], higher = better match
        confidence = edge["confidence"]
        total += alignment * confidence * importance
        weight_sum += importance
    if weight_sum == 0:
        return 0.0
    return total / weight_sum  # normalized to [-1, +1]
```

**Tiebreakers** (applied in order):
1. Higher total `mention_count` across matched attributes (more data = more trustworthy).
2. Higher `confidence` average.
3. Recency of reviews.

Products with no matching attributes at all are excluded, not scored zero — this prevents cold-start items from polluting results.

## 7. Domain Adaptation

Adding a new domain (e.g., running shoes) requires:

1. **Domain config file** (YAML):
```yaml
domain: running_shoe
categories: [road, trail, track, cross_training]
attributes:
  cushioning:
    synonyms: [cushion, plushness, softness, padding]
    spectrum: [minimal, moderate, maximal]
  responsiveness:
    synonyms: [energy_return, snap, springiness, bounce]
    spectrum: [dead, moderate, responsive, bouncy]
  stability:
    synonyms: [support, pronation_control, rigidity]
    spectrum: [neutral, mild_support, stability, motion_control]
  weight:
    synonyms: [heft, heaviness, lightness]
    spectrum: [ultralight, light, moderate, heavy]
constraints:
  - name: plate
    type: boolean
    synonyms: [carbon_plate, propulsion_plate]
  - name: stack_height_mm
    type: numeric
```

2. **Review data** as JSON with the standard fields:
```json
{"product_id": "shoe-001", "product_name": "Nike Vaporfly 3", "domain": "running_shoe", "review_text": "...", "source": "runrepeat"}
```

3. Run the ingestion pipeline. The ABSA model extracts aspects generically; the domain config handles normalization to canonical attributes.

No retraining required for the core pipeline. The attribute synonym dictionary is the primary adaptation lever. For domains with unusual jargon, a one-time pass with an LLM can bootstrap the synonym lists from a handful of sample reviews.

## 8. Pros and Cons

### Strengths
- **Fully explainable.** Every recommendation traces back to specific review sentences through graph edges. No black-box embeddings.
- **No cold-start problem for users.** The query IS the preference — no history needed.
- **Inspectable and debuggable.** You can visualize the graph, examine edge weights, and understand exactly why product A ranked above product B.
- **Domain-portable.** New domains need only a config file and data. The graph structure is universal.
- **Composable queries.** Boolean constraints (length >= 180cm, no carbon plate) are trivial graph filters, not awkward vector-space hacks.
- **Lightweight.** No GPU required for inference after initial ABSA extraction. Query resolution is pure graph traversal.

### Weaknesses
- **ABSA quality is the bottleneck.** If the extractor misidentifies aspects or sentiments, the entire graph is polluted. Garbage in, garbage out — and there is no embedding space to smooth over extraction errors.
- **Attribute ontology is manual.** Someone must define the attribute taxonomy and synonyms per domain. This is a one-time cost but non-trivial for unfamiliar domains.
- **Scales awkwardly.** NetworkX is single-threaded and in-memory. Not a concern at benchmark scale (50-200 products), but would need rethinking for production-scale catalogs.
- **Misses latent relationships.** A vector-space approach might discover that "playful" and "short turn radius" correlate without being told. The graph only knows relationships you explicitly model.
- **No semantic fuzziness.** If a user says "forgiving" and no attribute maps to it, the query returns nothing. Embedding-based systems degrade gracefully; graph systems fail hard on vocabulary gaps. **Mitigation:** a hybrid fallback is included (see below) — when graph-based attribute matching produces zero or very few results, the system falls back to embedding similarity between the raw query text and concatenated review text per product. This keeps the graph as the primary ranking mechanism but prevents zero-result failures.

## 8.1. Vocabulary Gap Fallback (Hybrid Embedding Rescue)

When query attribute matching returns fewer than `min_graph_matches` results (default: 2), the system engages a fallback that uses embedding similarity to prevent zero-result failures:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embedding_fallback(query_text: str, products: list[str], graph: nx.DiGraph,
                       top_k: int = 10) -> list[dict]:
    """Fall back to embedding similarity when graph matching fails."""
    # Concatenate all review text per product
    product_texts = {}
    for pid in products:
        reviews = [
            graph.nodes[rev].get("text", "")
            for rev in graph.successors(pid)
            if graph.nodes[rev].get("type") == "review"
        ]
        product_texts[pid] = " ".join(reviews)

    query_emb = model.encode(query_text)
    scored = []
    for pid, text in product_texts.items():
        if not text:
            continue
        prod_emb = model.encode(text)
        sim = float(query_emb @ prod_emb / (
            (query_emb @ query_emb) ** 0.5 * (prod_emb @ prod_emb) ** 0.5
        ))
        scored.append({"product_id": pid, "score": sim})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
```

The graph remains the primary ranking signal. The fallback only activates when the graph cannot resolve the query, ensuring that well-covered queries still get the full explainability benefit of graph traversal.

## 9. POC Scope

**Goal:** Ingest 50 ski reviews, build a graph, answer 3 sample queries with ranked results and explanations.

**Timeline:** One weekend.

**Data:** Hand-curated JSON file with ~50 reviews across ~10 ski models, in the standard format:

```json
{"product_id": "ski-001", "product_name": "Volkl Deacon 80", "domain": "ski", "review_text": "...", "source": "blister"}
```

### Minimal code sketch

```python
# poc.py
import networkx as nx
import json
import os
from dataclasses import dataclass

# --- Domain config (inline for POC) ---
SKI_ATTRIBUTES = {
    "stiffness": {"spectrum": [-1, 1]},
    "dampness":  {"spectrum": [-1, 1]},
    "agility":   {"spectrum": [-1, 1]},
    "edge_grip": {"spectrum": [-1, 1]},
}

# --- LLM-based ABSA via Ollama ---
def extract_aspects_llm(text: str, domain: str = "ski") -> list[tuple[str, float]]:
    """Use LLM to extract (canonical_attr, sentiment) pairs from review text."""
    import requests

    prompt = f"""Analyze this {domain} product review and extract aspect-sentiment pairs.
For each aspect mentioned, return the canonical attribute name and a sentiment score from -1.0 (very negative) to +1.0 (very positive).

Known attributes: {', '.join(SKI_ATTRIBUTES.keys())}
You may also identify new attributes not in this list.

Review: "{text}"

Return JSON array only, e.g.: [{{"attribute": "stiffness", "sentiment": 0.9}}, ...]
"""
    try:
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }, timeout=30)
        raw = resp.json().get("response", "[]")
        pairs = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(pairs, dict):
            pairs = pairs.get("aspects", pairs.get("results", []))
        return [(p["attribute"], float(p["sentiment"])) for p in pairs
                if "attribute" in p and "sentiment" in p]
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return []

# --- Graph construction ---
def build_graph(products: list[dict], reviews: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for p in products:
        G.add_node(p["id"], **p)
    for attr_name in SKI_ATTRIBUTES:
        G.add_node(f"attr-{attr_name}", type="attribute", name=attr_name)
    for rev in reviews:
        G.add_node(rev["id"], type="review", text=rev["text"])
        G.add_edge(rev["product_id"], rev["id"], type="REVIEWED_BY")
        for attr_name, sentiment in extract_aspects_llm(rev["text"]):
            G.add_edge(rev["id"], f"attr-{attr_name}", type="MENTIONS",
                       sentiment=sentiment, snippet=rev["text"][:120])
    # Aggregate into HAS_ATTRIBUTE edges
    for p in products:
        for attr_name in SKI_ATTRIBUTES:
            attr_id = f"attr-{attr_name}"
            mentions = [
                G.edges[e]["sentiment"]
                for e in G.in_edges(attr_id, data=False)
                if G.nodes[e[0]].get("type") == "review"
                and G.has_edge(p["id"], e[0])
            ]
            if mentions:
                G.add_edge(p["id"], attr_id, type="HAS_ATTRIBUTE",
                           score=sum(mentions)/len(mentions),
                           confidence=min(1.0, len(mentions)/5),
                           mention_count=len(mentions))
    return G

# --- Query resolution ---
def query(G: nx.DiGraph, query_text: str, top_k: int = 5) -> list[dict]:
    query_attrs = extract_aspects_llm(query_text)
    products = [n for n, d in G.nodes(data=True) if d.get("type") == "product"]
    scored = []
    for pid in products:
        total, w_sum = 0.0, 0.0
        for attr_name, desired_polarity in query_attrs:
            attr_id = f"attr-{attr_name}"
            edge = G.edges.get((pid, attr_id))
            if edge and edge.get("type") == "HAS_ATTRIBUTE":
                alignment = edge["score"] * (1 if desired_polarity > 0 else -1)
                total += alignment * edge["confidence"]
                w_sum += 1.0
        if w_sum > 0:
            scored.append({"product": G.nodes[pid]["name"],
                           "score": round(total / w_sum, 3),
                           "matched_attrs": len(query_attrs)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

if __name__ == "__main__":
    # Inline test data
    products = [
        {"id": "ski-001", "type": "product", "name": "Volkl Deacon 80", "domain": "ski"},
        {"id": "ski-002", "type": "product", "name": "Nordica Enforcer 94", "domain": "ski"},
    ]
    reviews = [
        {"id": "r1", "product_id": "ski-001", "text": "Incredibly stiff and damp. Great edge grip on ice."},
        {"id": "r2", "product_id": "ski-001", "text": "Very rigid, stable at speed, but not very agile."},
        {"id": "r3", "product_id": "ski-002", "text": "Playful and agile, moderate flex. A bit chattery on hardpack."},
        {"id": "r4", "product_id": "ski-002", "text": "Nimble ski, soft enough to be forgiving. Edge grip is decent."},
    ]
    G = build_graph(products, reviews)
    results = query(G, "stiff damp ski with great edge grip")
    for r in results:
        print(f"  {r['score']:+.3f}  {r['product']}")
```

### POC deliverables
1. `poc.py` — self-contained script as above, runnable with `python poc.py`.
2. `reviews.json` — 50 curated ski reviews across 10 models.
3. A brief results log showing ranked output for 3 test queries.

### Upgrade path from POC to production
- Replace inline data with SQLite-backed review store.
- Add FastAPI wrapper for the query endpoint.
- Add graph visualization endpoint (export to D3 or Cytoscape format).
- Fine-tune ABSA prompts per domain for higher extraction accuracy.

## 10. Common Interface

All 10 designs will be benchmarked against each other using a shared interface. This design will implement the following protocol:

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class RecommendationResult:
    product_id: str
    product_name: str
    score: float  # 0-1 normalized
    explanation: str
    matched_attributes: dict[str, float]  # attribute -> match strength

class Recommender(Protocol):
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None: ...
    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]: ...
```

### Mapping to this design

- **`ingest()`** maps to the ingestion pipeline (Section 4): text cleaning, LLM-based ABSA extraction, attribute normalization, and graph construction. The `domain` parameter selects the appropriate domain config for attribute canonicalization.
- **`query()`** maps to the query pipeline (Section 5): parse query into attribute targets, build query subgraph, filter candidates, score via graph matching, and return ranked results. When graph matching yields fewer than 2 results, the embedding fallback (Section 8.1) activates. Scores are normalized from the internal `[-1, +1]` range to `[0, 1]` for the common interface.
- **`RecommendationResult.explanation`** is populated from the `MENTIONS` edge snippets — the specific review sentences that support each attribute match.
- **`RecommendationResult.matched_attributes`** is populated from the `HAS_ATTRIBUTE` edge data, mapping each matched attribute name to its alignment score.
