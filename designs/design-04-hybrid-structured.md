# Design 04: Hybrid Structured + Vector Recommendation System

**Design philosophy:** Dual-track retrieval combining ABSA-extracted structured attributes with semantic vector embeddings, unified through a multi-signal ranking layer. This is the reference implementation of the one-pager's original vision.

---

## 1. Architecture Overview

The system operates two parallel data paths that converge at query time:

```
                         INGESTION
                            |
              Raw Reviews (any domain)
                            |
                    ABSA Extraction
                     /            \
            Structured              Semantic
            Attributes              Embeddings
               |                       |
           SQLite DB                ChromaDB
               |                       |
                \                     /
                 --- Query Engine ---
                        |
               Multi-Signal Ranker
                        |
                  Ranked Results
```

**Track 1 (Structured):** Reviews are decomposed into (aspect, opinion, sentiment_score) triples via ABSA. Aspects are normalized against a domain ontology and stored in SQLite with typed attribute columns. This enables hard filtering ("180cm+", "no plate") and precise attribute matching.

**Track 2 (Semantic):** Full review text and extracted attribute summaries are embedded into dense vectors via a sentence transformer. This captures latent preferences, subjective feel descriptors ("buttery", "poppy"), and cross-domain similarity that structured attributes miss.

**Convergence:** At query time, vector recall produces a broad candidate set; structured filters narrow it; a weighted scoring function fuses similarity, attribute match rate, and sentiment strength into a final ranking.

---

## 2. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| ABSA extraction | LLM via Ollama (Llama 3 8B) or Claude API; PyABSA as offline fallback | LLM handles novel domains without retraining; local Ollama avoids API costs for POC |
| NLP utilities | spaCy (en_core_web_sm) | Entity extraction, noun-phrase chunking, query parsing |
| Embeddings | BGE-small-en-v1.5 (via sentence-transformers) | Strong retrieval performance at 384-dim; small enough for local CPU inference |
| Vector index | ChromaDB | Built-in persistence, ID management, and metadata filtering; eliminates manual ID mapping required by FAISS |
| Structured store | SQLite (via sqlite3 stdlib) | Zero-config, single-file, adequate for millions of rows |
| Serving | FastAPI (optional, POC can be CLI) | Async, lightweight, easy to prototype |
| Config/ontology | YAML files per domain | Human-editable, version-controllable |

All components run locally. No cloud services required for the POC.

---

## 3. Data Model

### 3.1 SQLite Schema

```sql
CREATE TABLE products (
    id          TEXT PRIMARY KEY,   -- slug or UUID
    name        TEXT NOT NULL,
    domain      TEXT NOT NULL,      -- "ski", "running_shoe", "cookie"
    raw_meta    TEXT,               -- JSON blob of original product metadata
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  TEXT REFERENCES products(id),
    source      TEXT,               -- "blister", "realskiers", "reddit", etc.
    author      TEXT,
    text        TEXT NOT NULL,
    overall_sentiment REAL,         -- -1.0 to 1.0
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE aspects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id   INTEGER REFERENCES reviews(id),
    product_id  TEXT REFERENCES products(id),
    raw_aspect  TEXT NOT NULL,      -- original text span: "edge hold"
    norm_aspect TEXT NOT NULL,      -- ontology-normalized: "edge_grip"
    opinion     TEXT,               -- "incredible on ice"
    sentiment   REAL NOT NULL,      -- -1.0 to 1.0
    confidence  REAL DEFAULT 1.0
);

-- Aggregated product-level attribute view
CREATE VIEW product_attributes AS
SELECT
    product_id,
    norm_aspect,
    AVG(sentiment) AS avg_sentiment,
    COUNT(*) AS mention_count
FROM aspects
GROUP BY product_id, norm_aspect;

CREATE INDEX idx_aspects_product ON aspects(product_id);
CREATE INDEX idx_aspects_norm ON aspects(norm_aspect);
```

### 3.2 Vector Index

Each product gets one or more vectors stored in ChromaDB:

- **Product summary vector:** Embedded from a synthesized text blending product metadata and aggregated aspect opinions. Example input: `"Nordica Enforcer 100: stiff flex, excellent edge grip, damp vibration, moderate weight, all-mountain ski"`.
- **Per-review vectors (optional):** Individual review embeddings for fine-grained recall. ChromaDB stores these with `review_id` as document ID, eliminating the need for a separate ID map.

The vector and structured stores share `product_id` as the join key. ChromaDB metadata fields mirror key structured attributes for pre-filtering.

---

## 4. Ingestion Pipeline

```
[Raw Data]  -->  [1. Parse & Clean]  -->  [2. ABSA Extract]  -->  [3. Normalize]
                                                                        |
                                                          +-------------+-------------+
                                                          |                           |
                                                  [4a. Store Structured]     [4b. Embed & Index]
```

### Step 1: Parse & Clean
Strip HTML, normalize unicode, segment into sentences. SpaCy sentence segmentation handles run-on review text.

### Step 2: ABSA Extraction
Each review is sent through the extraction layer, which returns structured triples.

**Primary path (POC): LLM extraction via Ollama (Llama 3 8B) or Claude API.** Ollama is preferred for local/free operation; Claude API is the fallback when higher extraction quality is needed. **PyABSA is a noted fallback** for fully offline, batch-oriented scenarios but is not the primary path.

```python
EXTRACTION_PROMPT = """\
You are an aspect-based sentiment analysis engine for {domain} product reviews.

Extract every product attribute mentioned in the review below. For each attribute, return:
- "aspect": a short noun phrase naming the attribute (e.g., "flex", "edge hold", "weight")
- "opinion": the reviewer's exact or paraphrased opinion about it
- "sentiment": a float from -1.0 (very negative) to 1.0 (very positive)

Rules:
- Only extract attributes about the product itself, not about shipping, price, or the reviewer.
- If the reviewer is comparative ("stiffer than X"), still extract the aspect with sentiment relative to positive.
- Return a JSON array. No commentary outside the JSON.

Review:
{review_text}
"""

def extract_aspects(review_text: str, domain: str) -> list[dict]:
    """Returns [{"aspect": "flex", "opinion": "very stiff", "sentiment": 0.85}, ...]"""
    prompt = EXTRACTION_PROMPT.format(domain=domain, review_text=review_text)
    # Primary: Ollama (Llama 3 8B) with JSON mode
    return call_ollama(prompt, model="llama3:8b", format="json")
    # Fallback: call_claude(prompt, model="claude-sonnet-4-20250514")
    # Offline fallback: PyABSA ATEPC pipeline (no prompt needed)
```

The LLM path uses structured output (JSON mode) for reliability. For the POC, Ollama with Llama 3 8B provides good extraction quality at zero API cost. Claude API can be swapped in for higher accuracy on ambiguous reviews.

### Step 3: Normalize Aspects
Raw aspect strings are mapped to canonical ontology terms:

```python
def normalize_aspect(raw: str, domain: str, ontology: dict) -> str:
    """Map 'edge hold' -> 'edge_grip', 'turn initiation' -> 'turn_entry'."""
    # 1. Exact match in synonym table
    if raw.lower() in ontology["synonyms"]:
        return ontology["synonyms"][raw.lower()]
    # 2. Fuzzy match via embedding similarity against ontology terms
    best = find_closest_ontology_term(raw, ontology["terms"])
    if best.score > 0.82:
        return best.term
    # 3. Keep raw, flag for human review
    return f"_unmatched:{raw}"
```

### Step 4a: Store Structured
Insert into `products`, `reviews`, `aspects` tables. Upsert logic handles re-ingestion.

### Step 4b: Embed & Index
Build the product summary string from aggregated aspects, embed with BGE, add to ChromaDB:

```python
def build_and_store_embedding(product_id: str, db: sqlite3.Connection, collection: chromadb.Collection):
    rows = db.execute(
        "SELECT norm_aspect, AVG(sentiment), COUNT(*) FROM aspects "
        "WHERE product_id = ? GROUP BY norm_aspect ORDER BY COUNT(*) DESC",
        (product_id,)
    ).fetchall()
    summary = build_summary_text(product_id, rows)  # "stiff flex, great edge grip, ..."
    embedding = embed_model.encode(summary).tolist()

    # ChromaDB handles persistence and ID mapping automatically
    collection.upsert(
        ids=[product_id],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[{"domain": domain, "name": product_name}]
    )
```

---

## 5. Query Pipeline

```
[User Query] --> [Parse] --> [Expand] --> [Vector Recall] --> [Structured Filter] --> [Rerank] --> [Results]
```

### Step 1: Parse Query
SpaCy + heuristics decompose the query into:

```python
@dataclass
class ParsedQuery:
    semantic_text: str              # full query for embedding
    required_aspects: list[str]     # "stiff", "on-piste" -> normalized
    numeric_filters: list[Filter]   # "180cm+" -> Filter(attr="length", op=">=", val=180)
    negations: list[str]            # "no plate" -> ["plate"]
    domain_hint: str | None         # "ski" inferred from "carving ski"
```

Numeric filters are extracted via regex patterns (`\d+\s*(cm|mm|g|oz|km)\s*[+\-]?`). Negations are detected by dependency parsing ("no X", "without X", "not X").

### Step 2: Expand
Synonym expansion using the domain ontology. "Stiff" might expand to include "rigid", "firm" for broader recall.

### Step 3: Vector Recall
Embed the `semantic_text`, query ChromaDB for top-K candidates (K=50-100):

```python
query_vec = embed_model.encode(parsed.semantic_text).tolist()
results = collection.query(
    query_embeddings=[query_vec],
    n_results=100,
    where={"domain": parsed.domain_hint} if parsed.domain_hint else None
)
candidates = results["ids"][0]
similarities = [1 - d for d in results["distances"][0]]  # ChromaDB returns L2; convert to similarity
```

### Step 4: Structured Filter
Apply hard filters on the candidate set via SQL:

```python
def apply_filters(candidates: list[str], parsed: ParsedQuery, db) -> list[str]:
    placeholders = ",".join("?" * len(candidates))
    query = f"SELECT DISTINCT product_id FROM product_attributes WHERE product_id IN ({placeholders})"
    params = list(candidates)

    # Exclude negated aspects
    for neg in parsed.negations:
        query += " AND product_id NOT IN (SELECT product_id FROM aspects WHERE norm_aspect = ? AND sentiment > 0)"
        params.append(neg)

    return [r[0] for r in db.execute(query, params).fetchall()]
```

#### Negation handling: concrete example

Consider the query **"stiff ski, no rocker"**:

1. **Parse:** `ParsedQuery(semantic_text="stiff ski no rocker", required_aspects=["flex_stiff"], negations=["rocker"], ...)`

2. **Vector recall:** The full `semantic_text` including "no rocker" is embedded. Because embedding models do not reliably capture negation semantics ("no rocker" embeds close to "rocker"), vector recall may still return rockered skis. This is expected and handled by the structured filter.

3. **Structured filter:** The SQL exclusion clause removes products where `rocker` has positive sentiment:
   ```sql
   -- Products with rocker praised are excluded
   AND product_id NOT IN (
       SELECT product_id FROM aspects
       WHERE norm_aspect = 'rocker' AND sentiment > 0
   )
   ```
   This means a ski described as having "minimal rocker" (low positive sentiment) might be excluded, while a ski with "no rocker, full camber" (negative or absent sentiment on rocker) passes through.

4. **Scoring:** Negated aspects do not contribute to the attribute match or sentiment signals -- they only act as hard filters. A product that passes the negation filter is scored solely on its positive matches (e.g., how well it matches "stiff flex").

**Key design decision:** Negation is handled entirely in the structured path, not the vector path. This is intentional: embedding models are unreliable with negation, so we rely on the structured filter to enforce "must not have" constraints precisely.

### Step 5: Rerank
Multi-signal scoring (see next section) produces the final ordered list.

---

## 6. Ranking Strategy

Each candidate product receives a composite score from three signals:

```python
def score_product(
    product_id: str, parsed: ParsedQuery, vector_sim: float, db,
    weights: dict[str, float] | None = None
) -> tuple[float, dict[str, float]]:
    """Returns (final_score, breakdown) where breakdown maps signal names to their weighted contributions."""
    w = weights or {"vector": 0.4, "attribute": 0.35, "sentiment": 0.25}

    # Signal 1: Vector similarity (0-1), already computed
    s_vector = vector_sim

    # Signal 2: Attribute match rate
    product_aspects = get_product_aspects(product_id, db)  # {norm_aspect: avg_sentiment}
    matched_attrs = {a: product_aspects[a] for a in parsed.required_aspects if a in product_aspects}
    s_attr_match = len(matched_attrs) / max(len(parsed.required_aspects), 1)

    # Signal 3: Sentiment strength on matched aspects
    sentiments = list(matched_attrs.values())
    s_sentiment = (sum(sentiments) / len(sentiments)) if sentiments else 0.0
    s_sentiment = (s_sentiment + 1) / 2  # normalize from [-1,1] to [0,1]

    # Weighted combination
    final = w["vector"] * s_vector + w["attribute"] * s_attr_match + w["sentiment"] * s_sentiment

    breakdown = {
        "vector_similarity": s_vector,
        "attribute_match": s_attr_match,
        "sentiment_strength": s_sentiment,
        **{f"aspect:{k}": (v + 1) / 2 for k, v in matched_attrs.items()}
    }
    return final, breakdown
```

**Weight rationale and tuning guidance:**

The default weights (0.4 / 0.35 / 0.25) are starting-point heuristics, not empirically tuned values. The reasoning behind the defaults:

- **Vector similarity (0.4)** is the primary recall signal -- it captures holistic "feel" and handles vague queries well. It gets the highest weight because it is the most robust signal: it works even when the query has no explicit attribute requirements.
- **Attribute match (0.35)** ensures explicit user requirements are met. A product missing a stated requirement gets heavily penalized. This is nearly as important as vector similarity because users who specify attributes expect them to be honored.
- **Sentiment strength (0.25)** differentiates products that merely mention an attribute from those that excel at it. A ski that is "somewhat stiff" ranks below one that is "very stiff" when the user asks for stiff. It gets the lowest weight because it refines rather than drives the ranking.

**These weights should be tuned per domain.** Different domains have different query patterns: ski queries tend to be attribute-heavy (favoring higher `attribute` weight), while cookie queries may be more vibes-driven (favoring higher `vector` weight). Weights are configurable in the domain YAML:

```yaml
# ontologies/ski.yaml (add to existing file)
ranking_weights:
  vector: 0.35
  attribute: 0.40
  sentiment: 0.25
  # NOTE: must sum to 1.0; loader validates this
```

Without user feedback data for systematic tuning, start with the defaults and adjust manually based on result quality during POC evaluation.

---

## 7. Domain Adaptation

The system handles new domains via pluggable ontology files:

```yaml
# ontologies/ski.yaml
domain: ski
attributes:
  flex:
    type: ordinal
    scale: [soft, medium, stiff, very_stiff]
    synonyms: [stiffness, rigidity, firmness]
  edge_grip:
    type: sentiment
    synonyms: [edge hold, grip on ice, ice performance, hold on hardpack]
  turn_entry:
    type: sentiment
    synonyms: [turn initiation, engagement, how it starts turns]
  length:
    type: numeric
    unit: cm
  waist_width:
    type: numeric
    unit: mm
categories:
  terrain: [on-piste, off-piste, all-mountain, park, touring]
ranking_weights:
  vector: 0.35
  attribute: 0.40
  sentiment: 0.25
```

### Bootstrapping a New Domain

When entering a new domain (e.g., cookies), an LLM generates a draft ontology:

```python
def bootstrap_ontology(domain: str, sample_reviews: list[str]) -> dict:
    prompt = f"""Given these {domain} reviews, extract a product attribute taxonomy.
    For each attribute, provide: name, type (ordinal/sentiment/numeric/categorical),
    and common synonyms found in reviews.
    Reviews: {sample_reviews[:10]}"""
    draft = call_llm(prompt)
    # Human reviews and edits the YAML before it goes live
    return draft
```

This produces a starting ontology that a domain expert refines. The system degrades gracefully on unmatched aspects -- they still contribute to vector similarity even if they are not in the structured path.

---

## 8. Pros and Cons

### Pros
- **Explainable results.** Structured attributes let you show users *why* a product ranked highly ("matched: stiff flex (0.92), excellent edge grip (0.87)") rather than just a similarity score.
- **Hard filter support.** Numeric and categorical constraints ("180cm+", "no plate") are handled precisely, not approximately. Vector-only systems struggle with these.
- **Graceful degradation.** If ABSA misses an aspect, vector similarity still captures it. If the query is too vague for structured matching, vectors carry the ranking. Neither track is a single point of failure.
- **Domain portability.** The ontology-per-domain approach means adding a new product category is a YAML file and a batch re-ingestion, not a model retrain.
- **Local-first, low cost.** BGE-small runs on CPU. SQLite and ChromaDB have zero infrastructure cost. LLM calls are only needed at ingestion time (or can be replaced with PyABSA for fully offline operation).

### Cons
- **Ontology maintenance burden.** Each domain needs a curated attribute taxonomy. Bootstrapping helps but human curation is still required for quality. This is the single largest ongoing cost.
- **ABSA quality is the ceiling.** If extraction misses aspects or assigns wrong sentiment, structured matching suffers. LLM extraction is good but not perfect, especially for sarcasm, comparative statements, and implicit opinions.
- **Ingestion latency.** LLM-based ABSA is slow (~1-3 seconds per review). Batch ingestion of thousands of reviews requires rate limiting and patience, or a fallback to PyABSA.
- **Synchronization complexity.** Two data stores (SQLite + ChromaDB) must stay consistent. Deleting or updating a product requires coordinated updates. Not hard, but a source of bugs if neglected.
- **Weight tuning.** The ranking weights (0.4/0.35/0.25) are starting-point heuristics (see Section 6 for rationale). Optimal weights vary by domain and query type. Weights are configurable per domain in the YAML ontology; without user feedback data, tuning is manual.
- **Cold start per domain.** A new domain with few reviews and a draft ontology will produce mediocre results until enough data flows through.

---

## 9. POC Scope

### What the POC demonstrates
A CLI tool that ingests ski reviews from a JSON file, builds both indexes, and answers natural language queries with ranked, explained results.

### Minimal file structure

```
recommend/
  cli.py              # Entry point: ingest / query commands
  ingest/
    extractor.py       # ABSA extraction (LLM + PyABSA)
    normalizer.py      # Ontology-based aspect normalization
    pipeline.py        # Orchestrates parse -> extract -> store -> embed
  query/
    parser.py          # Query decomposition
    retriever.py       # Vector recall + structured filtering
    ranker.py          # Multi-signal scoring
  storage/
    db.py              # SQLite helpers
    vectors.py         # ChromaDB index management
  ontologies/
    ski.yaml
  data/
    sample_reviews.json
```

### POC code sketch: end-to-end query

```python
# cli.py (simplified)
import argparse
from ingest.pipeline import ingest_reviews
from query.parser import parse_query
from query.retriever import retrieve
from query.ranker import rank

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("ingest").add_argument("--file", required=True)
    q = sub.add_parser("query")
    q.add_argument("text", nargs="+")
    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_reviews(args.file, domain="ski")
    elif args.cmd == "query":
        query_text = " ".join(args.text)
        parsed = parse_query(query_text, domain="ski")
        candidates, similarities = retrieve(parsed, top_k=50)
        results = rank(candidates, similarities, parsed)
        for i, r in enumerate(results[:10], 1):
            print(f"{i}. {r.name} (score: {r.score:.3f})")
            for reason in r.explanations:
                print(f"   - {reason}")

if __name__ == "__main__":
    main()
```

### POC milestones
1. **Ingest 20-50 ski reviews** from a hand-curated JSON file. Verify ABSA extraction quality.
2. **Build both indexes.** Confirm SQLite has correct aspect triples; ChromaDB returns sensible nearest neighbors.
3. **Run 5 test queries** covering different patterns: attribute-heavy ("stiff, narrow, on-piste"), vibes-heavy ("playful all-mountain ski"), mixed with numerics ("180cm+ powder ski"), negation ("no rocker"), and vague ("good ski for intermediates").
4. **Evaluate and tune.** Compare results against expert intuition. Adjust ranking weights.

### Dependencies (pip)

```
sentence-transformers>=2.2.0   # BGE embeddings
chromadb>=0.4.0                # Vector index with persistence
spacy>=3.5.0                   # Query parsing
pyyaml>=6.0                    # Ontology files
httpx>=0.24.0                  # LLM API calls (Ollama / Claude)
```

---

## 10. Common Interface

All recommendation designs implement this common protocol so that benchmarking, CLI tooling, and future API layers work against a single contract:

```python
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RecommendationResult:
    product_id: str
    product_name: str
    score: float  # 0-1 normalized
    explanation: str
    matched_attributes: dict[str, float]


class Recommender(Protocol):
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Ingest product metadata and reviews into the system.

        Args:
            products: List of product dicts with at least 'id' and 'name' keys.
            reviews: List of review dicts with at least 'product_id' and 'text' keys.
            domain: Domain identifier (e.g., "ski", "running_shoe") for ontology lookup.
        """
        ...

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        """Query for product recommendations.

        Args:
            query_text: Natural language query (e.g., "stiff ski, no rocker").
            domain: Domain to search within.
            top_k: Number of results to return.

        Returns:
            List of RecommendationResult, sorted by score descending.
        """
        ...
```

### Implementation mapping for this design

The `ingest` method orchestrates the full ingestion pipeline (Section 4): parse, ABSA extract, normalize, store to SQLite, embed to ChromaDB. The `query` method orchestrates the query pipeline (Section 5): parse query, vector recall, structured filter, rerank, then builds `RecommendationResult` objects with explanations (see Section 11).

---

## 11. Explanation Generation

The design promises explainable results ("matched: stiff flex (0.92)") but the explanation string must be constructed concretely from the scoring breakdown. Here is the function:

```python
def build_explanation(
    product_name: str,
    breakdown: dict[str, float],
    parsed: ParsedQuery,
    weights: dict[str, float],
) -> str:
    """Build a human-readable explanation from the scoring breakdown.

    Args:
        product_name: Display name of the product.
        breakdown: Dict from score_product(), e.g.,
            {"vector_similarity": 0.78, "attribute_match": 1.0, "sentiment_strength": 0.91,
             "aspect:flex": 0.92, "aspect:edge_grip": 0.87}
        parsed: The parsed query, for listing negation filters applied.
        weights: The ranking weights used (for transparency).

    Returns:
        A multi-line explanation string.
    """
    lines = []

    # Matched aspects with their normalized sentiment scores
    aspect_lines = []
    for key, value in breakdown.items():
        if key.startswith("aspect:"):
            aspect_name = key.split(":", 1)[1].replace("_", " ")
            aspect_lines.append(f"  matched: {aspect_name} ({value:.2f})")
    if aspect_lines:
        lines.append("Attribute matches:")
        lines.extend(sorted(aspect_lines, key=lambda l: -float(l.split("(")[1].rstrip(")"))))

    # Signal summary
    lines.append(f"Signals: vector similarity {breakdown['vector_similarity']:.2f}, "
                 f"attribute match {breakdown['attribute_match']:.2f}, "
                 f"sentiment strength {breakdown['sentiment_strength']:.2f}")

    # Negation filters
    if parsed.negations:
        neg_str = ", ".join(n.replace("_", " ") for n in parsed.negations)
        lines.append(f"Excluded attributes confirmed absent: {neg_str}")

    return "\n".join(lines)
```

**Example output** for query "stiff ski, no rocker" matching the Nordica Enforcer 100:

```
Attribute matches:
  matched: flex (0.92)
  matched: edge grip (0.87)
Signals: vector similarity 0.78, attribute match 1.00, sentiment strength 0.91
Excluded attributes confirmed absent: rocker
```

This explanation string is stored directly in `RecommendationResult.explanation` and printed by the CLI.

---

## Summary

This design bets on the idea that recommendations work best when you combine "I know what you mean" (vectors) with "I know exactly what you asked for" (structured attributes). Vectors handle the fuzzy, subjective, hard-to-specify aspects of product feel. Structured data handles the precise, filterable, explainable aspects. The ranking layer blends both, weighted by confidence.

The main risk is ontology curation effort per domain. The main advantage is result explainability and precise constraint handling. For a POC, the ski domain alone is sufficient to validate the approach before generalizing.
