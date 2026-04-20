# Design #10: Ensemble / Meta-Learner (Learning to Rank)

**Design philosophy:** No single retrieval signal is universally best. BM25 catches keyword matches a vector search misses; embeddings capture vibes that structured filters cannot express; sentiment scores separate "mentioned" from "loved." This design treats each retrieval strategy as a **feature producer**, then trains a lightweight Learning to Rank (LTR) model to combine those signals into a final score. The meta-ranker learns, from a small amount of feedback data, how much to trust each signal for each domain and query type.

---

## 1. Architecture Overview

```
                        INGESTION
                           |
             Raw Reviews (any domain)
                           |
                   ABSA / LLM Extraction
                  /       |         \
          BM25 Index   Vector Index   Structured DB
         (rank_bm25)   (FAISS flat)   (SQLite)
                  \       |         /
                   --- stored independently ---

                      QUERY TIME
                           |
                    Parse Query Intent
                           |
            +-----------------------------+
            |  Parallel Retrieval Fanout  |
            +-----------------------------+
            |  BM25     |  Vector  | SQL  |
            |  Score    |  Score   | Match|
            +-----------------------------+
                           |
                  Feature Vector Assembly
            (per candidate query-product pair)
                           |
                    Meta-Ranker (LTR)
                      XGBoost / LR
                           |
                Score Normalization (sigmoid)
                           |
              Explanation (SHAP feature contributions)
                           |
                    Ranked Results
```

The architecture decomposes into three independent index-building pipelines (ingestion) and a query-time fanout that gathers signals from all three, builds a per-candidate feature vector, and feeds it into a trained ranker. The key insight is that index construction is simple and well-understood for each retrieval backend; the complexity lives only in the combiner.

---

## 2. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| BM25 index | **rank_bm25** | Pure Python, in-memory, actively maintained. Whoosh is effectively abandoned (last release 2015, no Python 3.12+ support). |
| Vector index | **FAISS** (flat index) | Fast, local, well-supported. Flat index for POC simplicity (no IVF training needed). Sentence-transformers for embedding generation. |
| Structured store | **SQLite** | Hard filters (size, category), aggregated attribute scores, product metadata. Also provides FTS5 as a fallback BM25-like scorer if needed. |
| Embedding model | **sentence-transformers** (`all-MiniLM-L6-v2`) | 384-dim, fast on CPU, good general-purpose quality. |
| ABSA / extraction | **LLM via Ollama or API** (llama3 8B, phi3, or cloud provider) | Extract (aspect, opinion, sentiment) triples from reviews. Supports both local Ollama and remote API endpoints (OpenAI-compatible, Anthropic) for flexibility. Falls back to spaCy + rules for speed. |
| Meta-ranker | **XGBoost** (primary), **scikit-learn LogisticRegression** (baseline) | XGBoost handles feature interactions well and trains in seconds on small datasets. LR is the interpretable fallback. |
| Score normalization | **Sigmoid** on XGBoost output | XGBoost ranker outputs unbounded scores; sigmoid maps them to 0-1 for the common interface. |
| Explanation engine | **SHAP** (via `xgboost` built-in or `shap` library) | Per-prediction feature contributions for human-readable explanations. |
| Query parser | **spaCy** + regex + optional LLM | Extract hard constraints, soft preferences, negations. |
| Evaluation | **scikit-learn metrics**, custom NDCG/MRR | Standard IR metrics for ranking quality. |

Total local footprint: approximately 300-500 MB RAM for a 10K-product corpus with all three indices loaded.

---

## 3. Data Model

### Product Record (SQLite)

```sql
CREATE TABLE products (
    id          TEXT PRIMARY KEY,
    domain      TEXT NOT NULL,         -- 'ski', 'running_shoe', 'cookie', ...
    name        TEXT NOT NULL,
    brand       TEXT,
    category    TEXT,                   -- 'carving', 'all-mountain', 'trail', ...
    metadata    JSON                    -- domain-specific: length_cm, weight_g, etc.
);

CREATE TABLE reviews (
    id          TEXT PRIMARY KEY,
    product_id  TEXT REFERENCES products(id),
    text        TEXT NOT NULL,
    rating      REAL,
    source      TEXT
);

CREATE TABLE extracted_attributes (
    id          INTEGER PRIMARY KEY,
    review_id   TEXT REFERENCES reviews(id),
    product_id  TEXT,
    aspect      TEXT NOT NULL,          -- normalized: 'stiffness', 'cushion', 'flavor'
    opinion     TEXT,                   -- raw opinion phrase: 'very stiff', 'too soft'
    sentiment   REAL NOT NULL,          -- -1.0 to 1.0
    confidence  REAL DEFAULT 1.0
);

CREATE INDEX idx_attr_product ON extracted_attributes(product_id, aspect);
```

### Aggregated Product Attributes (materialized view / computed table)

```sql
CREATE TABLE product_attribute_scores (
    product_id  TEXT,
    aspect      TEXT,
    avg_sentiment   REAL,   -- mean sentiment across all reviews for this aspect
    mention_count   INTEGER,
    confidence      REAL,   -- weighted by review count and extraction confidence
    PRIMARY KEY (product_id, aspect)
);
```

### Feature Vector (query time, in-memory)

```python
@dataclass
class CandidateFeatures:
    """Feature vector for a single (query, product) pair, fed to the meta-ranker."""
    # Retrieval scores (raw signals)
    bm25_score: float           # BM25 relevance of query against product's review corpus
    vector_similarity: float    # Cosine similarity: query embedding vs. product embedding
    attribute_match_rate: float # Fraction of query-requested attributes this product matches
    attribute_sentiment_avg: float  # Mean sentiment on matched attributes
    negation_violation: float   # 1.0 if product matches a negated constraint, else 0.0

    # Product-level features
    review_count: int
    avg_rating: float
    attribute_coverage: float   # How many domain aspects have extracted data

    # Query-product interaction features
    hard_filter_pass: bool      # Did product pass all hard constraints (size, category)?
    sentiment_gap: float        # Difference between desired polarity and actual sentiment
    domain_match: bool          # Query domain matches product domain
```

This feature vector is what makes the design flexible: adding a new retrieval signal means adding one more float to the vector and retraining.

---

## 4. Ingestion Pipeline

```
Raw Reviews
    |
    v
[1] Text Cleaning & Dedup
    |
    v
[2] ABSA Extraction (LLM via Ollama or API, or rule-based fallback)
    |  -> (aspect, opinion, sentiment, confidence) triples
    |  -> stored in extracted_attributes
    |
    v
[3] Parallel Index Building (independent, can run concurrently)
    |
    +---> [3a] BM25 Index
    |     Concatenate all reviews per product into a single "review document."
    |     Tokenize and build in-memory index via rank_bm25 (BM25Okapi).
    |
    +---> [3b] Vector Index
    |     Generate per-product embedding by mean-pooling sentence embeddings
    |     across all reviews (weighted by rating or confidence).
    |     Store in FAISS with product ID mapping.
    |
    +---> [3c] Structured Attribute Aggregation
          Compute product_attribute_scores from extracted_attributes.
          Standard SQL GROUP BY + AVG.
```

### Incremental Updates

When new reviews arrive, each pipeline processes only the delta:
- **BM25:** Rebuild the product's concatenated document, update index entry.
- **Vector:** Recompute the product embedding (incremental mean update).
- **Structured:** Re-aggregate that product's attribute scores.

The meta-ranker does not need retraining for new products; it operates on features, not product IDs.

---

## 5. Query Pipeline

```python
def recommend(query: str, domain: str, top_k: int = 10) -> list[RankedProduct]:
    # Step 1: Parse the query
    parsed = parse_query(query)
    # -> hard_constraints: {length_cm: ">= 180", category: "carving"}
    # -> soft_preferences: [("stiffness", +1), ("on-piste", +1)]
    # -> negations: ["plate"]
    # -> free_text: "stiff on-piste carving ski 180cm+"

    # Step 2: Hard-filter candidate set from SQLite
    candidates = sql_filter(parsed.hard_constraints, domain)
    # Typically narrows from 10K to 50-500 products

    # Step 3: Parallel retrieval scoring over candidate set
    bm25_scores = bm25_index.score(parsed.free_text, candidates)
    vector_scores = faiss_index.score(embed(parsed.free_text), candidates)
    attr_scores = attribute_matcher.score(parsed.soft_preferences, candidates)

    # Step 4: Assemble feature vectors
    features = []
    for pid in candidates:
        fv = build_feature_vector(
            query=parsed,
            product_id=pid,
            bm25=bm25_scores[pid],
            vector=vector_scores[pid],
            attr=attr_scores[pid],
        )
        features.append(fv)

    # Step 5: Meta-ranker prediction
    X = np.array([fv.to_array() for fv in features])
    scores = meta_ranker.predict(X)  # pointwise scores

    # Step 6: Sort and return top-k
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [build_result(candidates[i], scores[i]) for i in ranked_indices]
```

The hard-filter step is critical for keeping the fanout manageable. Without it, scoring every product across three backends would be expensive. For a 10K-product catalog, a hard filter that retains 200 candidates keeps query latency well under 100ms.

---

## 6. Ranking Strategy

### Learning to Rank Formulation

The meta-ranker is trained using an **LTR objective** over (query, product, relevance) triples.

**Pointwise approach (simplest, POC default):** Treat each (query, product) pair as an independent regression problem. Train XGBoost or logistic regression to predict a relevance score (0-1). Loss: MSE or log-loss.

**Pairwise approach (better ranking quality):** For each query, form pairs (product_a, product_b) where one is more relevant. Train on the pair ordering. XGBoost's `rank:pairwise` objective implements LambdaMART, which is the industry standard.

**Listwise approach (best, needs more data):** Optimize NDCG directly over the full ranked list per query. XGBoost's `rank:ndcg` objective supports this.

### Where Training Data Comes From

This is the hardest part of the design. Strategies, ordered by practicality:

1. **Synthetic judgments from LLM (POC approach):** Given a query and a product with its extracted attributes, ask an LLM to rate relevance 0-3. Supports both local (Ollama) and remote API (OpenAI-compatible, Anthropic) providers. The POC pre-generates a fixed training set and saves it to disk (`training_data/synthetic_judgments.json`) so results are reproducible across runs. Details:
   - **Query set:** 40-50 diverse queries per domain covering different intent types (specific product needs, comparative queries, vague preferences, constraint-heavy queries).
   - **Products per query:** Each query is judged against 20-30 candidate products (stratified: some clearly relevant, some marginal, some irrelevant) for a total of ~1,000-1,500 judgment triples per domain.
   - **Prompt template:** See Section 9 code sketch. The prompt includes the query, product name, top-5 extracted attributes with sentiment scores, product metadata (category, brand), and review count. The LLM returns a JSON object with a 0-3 relevance score and a brief reason.
   - **Reproducibility:** The generated dataset is saved with a hash of the input corpus so staleness can be detected. Re-generation is triggered manually, not automatically.
2. **Click-through / interaction data:** Once deployed, log which results users click, add to cart, or dwell on. Implicit relevance signal.
3. **Expert annotations:** Small batch of human-labeled query-product pairs. 50-100 judgments per domain bootstraps a reasonable ranker.
4. **Heuristic labels:** Use a weighted combination of the raw retrieval scores as pseudo-labels, then train the ranker to refine them. Circular, but effective for initial calibration.

### Feature Importance as Interpretability

XGBoost gives feature importance scores out of the box. This tells you which signals matter most per domain, effectively learning that "for skis, attribute matching dominates; for cookies, sentiment scoring matters more."

### SHAP-Based Per-Result Explanations

Beyond global feature importance, the system uses SHAP (SHapley Additive exPlanations) values to generate per-prediction explanations. For each recommended product, the SHAP values quantify how much each feature contributed to that product's final score.

Example explanation output:

> Ranked #1 because: high attribute match (0.45 contribution), strong review sentiment (0.30 contribution), good vector similarity (0.20 contribution), above-average review count (0.05 contribution).

This is computed at query time using XGBoost's built-in `predict(X, pred_contribs=True)` (tree SHAP, runs in microseconds per candidate, no additional cost). The raw SHAP values are mapped to human-readable feature names and sorted by absolute contribution to produce the explanation string stored in `RecommendationResult.explanation`.

### Score Normalization

XGBoost's `rank:pairwise` objective outputs unbounded scores (positive or negative, depending on relative ordering). To satisfy the common interface's 0-1 score contract, the raw scores are passed through a sigmoid function:

```python
def normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Sigmoid normalization of XGBoost ranker output to 0-1 range."""
    return 1.0 / (1.0 + np.exp(-raw_scores))
```

This preserves the ranking order while providing a bounded, interpretable confidence score. For the POC, sigmoid is preferred over min-max normalization because it is stable across different query result sets (min-max would rescale per query, making scores incomparable across queries).

---

## 7. Domain Adaptation

### Per-Domain Feature Sets

The core feature vector is domain-agnostic (BM25 score, vector similarity, match rate are universal). Domain-specific features are appended:

```python
DOMAIN_FEATURES = {
    "ski": ["length_match", "binding_compatibility", "terrain_score"],
    "running_shoe": ["stack_height_match", "drop_match", "pronation_score"],
    "cookie": ["dietary_match", "texture_score"],
}
```

These are computed from structured metadata and extracted attributes specific to each domain.

### Adaptation Strategies

**Strategy A: One ranker per domain.** Train separate XGBoost models. Simple, clean, no cross-domain contamination. Works well with 50+ labeled queries per domain.

**Strategy B: Shared ranker with domain feature.** Add `domain` as a categorical feature. The tree model learns domain-specific splits automatically. Better when you have few queries per domain but many domains.

**Strategy C: Transfer learning.** Pre-train on a data-rich domain (e.g., skis with abundant reviews), then fine-tune on a new domain with minimal data. With XGBoost this means initializing from an existing model's structure and continuing training (`xgb_model` parameter).

### Bootstrapping a New Domain

1. Run ABSA extraction on the new domain's reviews to populate indices.
2. Use LLM-generated synthetic judgments (Strategy 1 from Section 6) to create 100-200 training pairs.
3. Train with Strategy B (shared ranker) or Strategy A if there is enough data.
4. The system produces reasonable results within minutes of ingesting a new corpus.

---

## 8. Pros and Cons

### Strengths

- **Most flexible ranking of any design.** Adding a new signal (e.g., price popularity, freshness, editorial boost) is one feature column and a retrain.
- **Interpretable.** XGBoost feature importance reveals what drives rankings. You can explain "this ski ranked first because it scored highest on attribute match and sentiment, even though BM25 ranked it fifth."
- **Graceful degradation.** If one index is missing or stale, the ranker relies on the remaining signals. No single point of failure in retrieval quality.
- **Industry-proven pattern.** LTR with gradient-boosted trees is how Google, Bing, Airbnb, and most production search engines rank results. This is not experimental.
- **Fast inference.** XGBoost prediction over 500 candidates with 12 features takes under 1ms. The bottleneck is retrieval, not ranking.
- **Domain transfer is natural.** Feature vectors are a universal interface; the model learns domain-specific weighting.

### Weaknesses

- **Needs training data.** Even a small ranker needs labeled (query, product, relevance) triples. Mitigated by LLM-synthetic judgments, but that adds a bootstrap dependency.
- **More moving parts than simpler designs.** Three indices, an extraction pipeline, a query parser, and a trained model. More operational surface area than Design 05 (SQL-first) or Design 02 (pure embedding). Mitigated for the POC by using only in-memory/embedded components (rank_bm25, FAISS flat, SQLite) with no external services.
- **Training/serving skew risk.** If extraction quality changes (new LLM version, different prompt), feature distributions shift and the ranker's learned weights may become stale. Requires monitoring.
- **Cold-start on new domains is weaker than embedding-only.** Design 02 can produce reasonable results with zero configuration; this design needs at minimum synthetic training data.
- **Feature engineering is manual.** Deciding what features to extract and how to compute them requires thought per domain. Partially automated by using universal signals (BM25, cosine similarity).

### Compared to Other Designs

| vs. Design | Advantage | Disadvantage |
|---|---|---|
| #01 Graph | Better ranking quality with data; simpler query model | Loses explicit relationship traversal |
| #02 Pure Embedding | Multi-signal beats single-signal; interpretable | Slower cold-start; more components |
| #03 LLM Judge | Cheaper at scale (no LLM per query); deterministic | Loses LLM's contextual judgment nuance |
| #04 Hybrid Structured | Learned weighting > hand-tuned weighting | Needs training data that #04 doesn't |
| #05 SQL-First | Handles fuzzy queries; learns from feedback | More complex; SQL is simpler to operate |

---

## 9. POC Scope

### Goal

Demonstrate end-to-end LTR ranking over a small ski review corpus. Show that the trained ranker outperforms any single retrieval signal alone.

### Minimal Components (Simplified for POC)

The POC focuses on proving the LTR concept works with minimal infrastructure. No web server, no neural ranker option, no disk-based search index. Three components only: `rank_bm25` (in-memory), FAISS flat index, and SQLite.

1. **Ingest 50-100 ski products** with reviews (scraped or synthetic).
2. **Build all three indices:** `rank_bm25` (in-memory BM25Okapi), FAISS flat index (sentence-transformers), SQLite (extracted attributes + metadata).
3. **Generate synthetic training data:** Pre-generate ~1,000 (query, product, relevance) triples via LLM (Ollama or API). Save to `training_data/synthetic_judgments.json` for reproducibility.
4. **Train XGBoost ranker** on the assembled feature vectors.
5. **Score normalization:** Apply sigmoid to XGBoost output for 0-1 bounded scores.
6. **Explanation generation:** Use SHAP values from XGBoost to build per-result explanations.
7. **Query CLI** that shows ranked results with scores, explanations, and per-signal breakdowns.

### Code Sketch: Meta-Ranker Training

```python
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GroupKFold

def train_meta_ranker(
    features: np.ndarray,    # (n_pairs, n_features)
    labels: np.ndarray,      # relevance scores 0-3
    query_ids: np.ndarray,   # group identifier per query
) -> xgb.XGBRanker:
    """Train an LTR model with pairwise loss."""

    ranker = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        tree_method="hist",
    )

    # XGBRanker needs group sizes: how many candidates per query
    unique_qids, group_sizes = np.unique(query_ids, return_counts=True)

    ranker.fit(
        features, labels,
        group=group_sizes,
        verbose=True,
    )

    return ranker


def evaluate_ranker(ranker, features, labels, query_ids):
    """Compute NDCG@10 per query."""
    from sklearn.metrics import ndcg_score

    scores = ranker.predict(features)
    unique_qids = np.unique(query_ids)
    ndcgs = []

    for qid in unique_qids:
        mask = query_ids == qid
        true = labels[mask].reshape(1, -1)
        pred = scores[mask].reshape(1, -1)
        if true.shape[1] > 1:
            ndcgs.append(ndcg_score(true, pred, k=10))

    return np.mean(ndcgs)
```

### Code Sketch: Feature Assembly

```python
from dataclasses import dataclass, astuple
from rank_bm25 import BM25Okapi
import faiss
import numpy as np

class FeatureAssembler:
    """Assembles per-candidate feature vectors from multiple retrieval signals."""

    def __init__(self, bm25_index, faiss_index, db, embed_fn):
        self.bm25 = bm25_index
        self.faiss = faiss_index
        self.db = db
        self.embed_fn = embed_fn

    def build_features(self, parsed_query, candidate_ids: list[str]) -> np.ndarray:
        query_text = parsed_query.free_text
        query_embedding = self.embed_fn(query_text)

        # Batch retrieval scores
        bm25_scores = self._bm25_scores(query_text, candidate_ids)
        vec_scores = self._vector_scores(query_embedding, candidate_ids)

        rows = []
        for i, pid in enumerate(candidate_ids):
            attr = self._attribute_features(parsed_query, pid)
            product = self._product_features(pid)
            row = [
                bm25_scores[i],
                vec_scores[i],
                attr["match_rate"],
                attr["sentiment_avg"],
                attr["negation_violation"],
                product["review_count"],
                product["avg_rating"],
                product["attribute_coverage"],
                float(attr["hard_filter_pass"]),
                attr["sentiment_gap"],
            ]
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    def _bm25_scores(self, query_text, candidate_ids):
        tokenized = query_text.lower().split()
        all_scores = self.bm25.get_scores(tokenized)
        # Map back to candidate subset
        return [all_scores[self._pid_to_idx(pid)] for pid in candidate_ids]

    def _vector_scores(self, query_embedding, candidate_ids):
        q = np.array([query_embedding], dtype=np.float32)
        D, I = self.faiss.search(q, k=len(candidate_ids))
        # Build a score map from FAISS results
        score_map = dict(zip(I[0], D[0]))
        return [score_map.get(self._pid_to_idx(pid), 0.0) for pid in candidate_ids]

    def _attribute_features(self, parsed_query, product_id):
        """Score how well a product's extracted attributes match query preferences."""
        prefs = parsed_query.soft_preferences  # [(aspect, polarity), ...]
        negs = parsed_query.negations           # [aspect, ...]

        scores = self.db.get_attribute_scores(product_id)
        matched = 0
        sentiments = []

        for aspect, desired_polarity in prefs:
            if aspect in scores:
                matched += 1
                sentiments.append(scores[aspect]["avg_sentiment"])

        neg_violation = any(a in scores and scores[a]["mention_count"] > 2 for a in negs)

        match_rate = matched / max(len(prefs), 1)
        sentiment_avg = np.mean(sentiments) if sentiments else 0.0
        sentiment_gap = abs(1.0 - sentiment_avg) if sentiments else 1.0

        return {
            "match_rate": match_rate,
            "sentiment_avg": sentiment_avg,
            "negation_violation": float(neg_violation),
            "hard_filter_pass": not neg_violation,
            "sentiment_gap": sentiment_gap,
        }

    def _product_features(self, product_id):
        meta = self.db.get_product_meta(product_id)
        return {
            "review_count": meta["review_count"],
            "avg_rating": meta["avg_rating"],
            "attribute_coverage": meta["attribute_coverage"],
        }
```

### Code Sketch: Synthetic Training Data Generation

```python
import json
import hashlib
from pathlib import Path
from abc import ABC, abstractmethod

# --- LLM Provider Abstraction ---

class LLMProvider(ABC):
    """Common interface for LLM providers used in extraction and judgment generation."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str: ...

class OllamaProvider(LLMProvider):
    """Local LLM via Ollama REST API."""
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "") -> str:
        import requests
        resp = requests.post(f"{self.base_url}/api/generate", json={
            "model": self.model, "prompt": prompt, "system": system, "stream": False
        })
        return resp.json()["response"]

class APIProvider(LLMProvider):
    """Remote LLM via OpenAI-compatible or Anthropic API."""
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", api_key: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str, system: str = "") -> str:
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model=self.model, max_tokens=256,
                system=system, messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        else:  # OpenAI-compatible
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(model=self.model, messages=messages, max_tokens=256)
            return resp.choices[0].message.content

# --- Judgment Generation ---

JUDGMENT_PROMPT = """You are evaluating product relevance for a search query.

Query: {query}
Product: {product_name}
Brand: {brand}
Category: {category}
Review count: {review_count}
Top attributes from reviews (aspect: avg_sentiment):
{attributes}

Rate relevance on a scale of 0-3:
  0 = completely irrelevant
  1 = marginally relevant (partially matches but missing key aspects)
  2 = relevant, good match (matches most criteria)
  3 = excellent match, exactly what was asked for

Respond with JSON only: {{"score": <int>, "reason": "<one sentence>"}}"""

TRAINING_DATA_DIR = Path("training_data")

def generate_synthetic_judgments(
    queries: list[str],
    products: list[dict],
    llm: LLMProvider,
    products_per_query: int = 25,
    output_path: Path = TRAINING_DATA_DIR / "synthetic_judgments.json",
) -> list[dict]:
    """Generate and persist (query, product, relevance) training triples via LLM.

    For the POC, this pre-generates a fixed training set saved to disk for
    reproducibility. Expects ~40-50 queries and selects ~25 products per query
    (stratified sample) for a total of ~1,000-1,250 judgments.
    """
    judgments = []

    for query in queries:
        # Stratified sample: pick a mix of candidates per query
        sampled = _stratified_sample(products, products_per_query)

        for product in sampled:
            attr_str = "\n".join(
                f"  {a['aspect']}: {a['avg_sentiment']:.2f}"
                for a in product.get("top_attributes", [])[:5]
            )
            prompt = JUDGMENT_PROMPT.format(
                query=query,
                product_name=product["name"],
                brand=product.get("brand", "unknown"),
                category=product.get("category", "unknown"),
                review_count=product.get("review_count", 0),
                attributes=attr_str or "  (no attributes extracted)",
            )
            response = llm.generate(prompt)
            parsed = json.loads(response)
            judgments.append({
                "query": query,
                "product_id": product["id"],
                "relevance": parsed["score"],
                "reason": parsed.get("reason", ""),
            })

    # Persist for reproducibility
    corpus_hash = hashlib.sha256(
        json.dumps(sorted([p["id"] for p in products])).encode()
    ).hexdigest()[:12]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "corpus_hash": corpus_hash,
        "query_count": len(queries),
        "products_per_query": products_per_query,
        "total_judgments": len(judgments),
        "judgments": judgments,
    }, indent=2))

    return judgments

def _stratified_sample(products, n):
    """Sample n products with a mix of categories to ensure diversity."""
    import random
    if len(products) <= n:
        return products
    # Group by category, round-robin pick
    by_cat = {}
    for p in products:
        by_cat.setdefault(p.get("category", "other"), []).append(p)
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
```

### POC Success Criteria

- NDCG@10 of the trained ranker exceeds the best single signal (BM25, vector, or attribute match alone) by at least 10%.
- Feature importance analysis shows that different signals matter for different query types.
- Each result includes a SHAP-derived explanation with per-feature contributions.
- All scores are normalized to 0-1 via sigmoid (common interface contract).
- Synthetic training data is saved to disk and produces identical rankings on reload.
- End-to-end latency under 200ms for a 1K-product catalog on a laptop.
- A new domain (e.g., cookies) can be bootstrapped and return reasonable results within 30 minutes of receiving review data.

### POC File Structure

```
recommend/
  ensemble_ltr/
    ingest.py          # Ingestion pipeline: extraction + index building
    indices.py         # rank_bm25, FAISS flat, SQLite wrappers
    features.py        # FeatureAssembler
    ranker.py          # XGBoost training + inference + score normalization
    explainer.py       # SHAP-based explanation generation
    query.py           # Query parsing + full pipeline
    llm_provider.py    # LLM abstraction (Ollama + API providers)
    synthetic.py       # LLM-based training data generation + persistence
    evaluate.py        # NDCG, MRR, ablation studies
    recommend.py       # Common Recommender interface implementation
    cli.py             # Interactive query interface
    config.py          # Domain configs, model paths, LLM provider settings
  training_data/
    synthetic_judgments.json  # Pre-generated, reproducible training set
```

---

## 10. Common Interface

All recommender designs implement a shared protocol so they can be benchmarked interchangeably.

### Data Types

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class RecommendationResult:
    product_id: str
    product_name: str
    score: float  # 0-1 normalized (sigmoid of XGBoost output)
    explanation: str  # SHAP-derived, e.g. "high attribute match (0.45), strong sentiment (0.30)"
    matched_attributes: dict[str, float]  # aspect -> contribution score
```

### Protocol

```python
class Recommender(Protocol):
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Ingest product catalog and reviews, building all internal indices.

        Args:
            products: List of product dicts with keys: id, name, brand, category, metadata.
            reviews: List of review dicts with keys: id, product_id, text, rating, source.
            domain: Domain identifier (e.g., 'ski', 'running_shoe', 'cookie').
        """
        ...

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        """Return ranked recommendations for a natural-language query.

        Args:
            query_text: Free-text query (e.g., "stiff carving ski for advanced skiers, 180cm+").
            domain: Domain to search within.
            top_k: Maximum number of results to return.

        Returns:
            List of RecommendationResult, sorted by score descending, scores in [0, 1].
        """
        ...
```

### Implementation for This Design

```python
import numpy as np

class EnsembleLTRRecommender:
    """Recommender implementation using ensemble LTR (Design #10)."""

    def __init__(self, llm: LLMProvider, config: dict | None = None):
        self.llm = llm
        self.config = config or {}
        self.indices = None       # Built during ingest
        self.meta_ranker = None   # Trained after synthetic data generation
        self.assembler = None     # FeatureAssembler

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        # 1. Run ABSA extraction via self.llm
        extracted = self._extract_attributes(reviews)

        # 2. Build indices (all in-memory for POC)
        self.indices = {
            "bm25": self._build_bm25(products, reviews),
            "faiss": self._build_faiss(products, reviews),
            "sqlite": self._build_sqlite(products, extracted, domain),
        }
        self.assembler = FeatureAssembler(
            self.indices["bm25"], self.indices["faiss"], self.indices["sqlite"], self._embed
        )

        # 3. Train meta-ranker (load saved judgments or generate new ones)
        self._train_ranker(products, domain)

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        parsed = parse_query(query_text)
        candidates = self.indices["sqlite"].filter(parsed.hard_constraints, domain)

        features = self.assembler.build_features(parsed, candidates)
        raw_scores = self.meta_ranker.predict(features)

        # Sigmoid normalization: unbounded XGBoost scores -> [0, 1]
        scores = 1.0 / (1.0 + np.exp(-raw_scores))

        # SHAP explanations
        shap_values = self.meta_ranker.predict(
            xgb.DMatrix(features), pred_contribs=True
        )

        ranked_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in ranked_idx:
            explanation, matched = self._build_explanation(shap_values[i])
            results.append(RecommendationResult(
                product_id=candidates[i],
                product_name=self.indices["sqlite"].get_name(candidates[i]),
                score=float(scores[i]),
                explanation=explanation,
                matched_attributes=matched,
            ))
        return results

    def _build_explanation(self, shap_row: np.ndarray) -> tuple[str, dict[str, float]]:
        """Convert SHAP values into a human-readable explanation string."""
        feature_names = [
            "BM25 relevance", "vector similarity", "attribute match",
            "review sentiment", "negation penalty", "review count",
            "avg rating", "attribute coverage", "hard filter", "sentiment gap",
        ]
        contributions = sorted(
            zip(feature_names, shap_row[:-1]),  # last element is bias
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        parts = [f"{name} ({val:+.2f})" for name, val in contributions if abs(val) > 0.01]
        explanation = "Ranked here because: " + ", ".join(parts[:4])
        matched = {name: float(val) for name, val in contributions if val > 0.01}
        return explanation, matched
```
