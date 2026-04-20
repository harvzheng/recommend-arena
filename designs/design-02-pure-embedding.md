# Design #2: Pure Embedding / Vector-First Recommendation

## 1. Architecture Overview

This design treats the recommendation problem as **semantic retrieval**: encode everything (reviews, product descriptions, user queries) into a shared dense vector space, then let cosine similarity do the matching. The central bet is that modern embedding models, trained on massive corpora, already understand the latent structure of language well enough to capture domain-specific attribute semantics -- "stiff" vs "forgiving," "responsive" vs "cushy" -- without explicit attribute extraction.

### Core Flow

```
User Query (NL)
    |
    v
Query Encoder --> query vectors
    |
    v
Vector Index (product representations built from reviews)
    |
    v
Top-K retrieval --> reranking --> ranked results
```

There is no structured attribute schema. No ABSA step. Products are represented as collections of embedding vectors derived from their reviews. Matching is purely geometric: a query like "stiff, damp carving ski for hardpack" lands near review passages that describe those characteristics.

### Why This Might Work

Review language IS the attribute space. When reviewers write "this ski is a damp, locked-in rail on hardpack," they are expressing exactly the attributes a buyer cares about. An embedding model trained on enough text will place that passage near the query "damp carving ski for hardpack" without anyone ever defining "damp" as a structured attribute.

### Why It Might Not

Embeddings encode similarity, not magnitude. "Very stiff" and "slightly stiff" might embed close together because they share semantics. This design must address ordinal ranking head-on (see Section 6).


## 2. Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Embedding model** | `BAAI/bge-large-en-v1.5` via `sentence-transformers` | Strong MTEB scores, single-vector simplicity, sufficient for POC |
| **Upgrade path** | ColBERTv2 (Stanford) via `colbert-ai` | Late interaction preserves token-level granularity; consider post-POC if single-vector retrieval proves insufficient |
| **Vector store** | ChromaDB (`chromadb` Python package, in-process) | Simple Python-native store, consistent with other designs, no external services |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight cross-encoder for top-K reranking |
| **Orchestration** | Plain Python, no framework | POC simplicity; FastAPI for optional API layer |
| **Tokenization/chunking** | `tiktoken` + custom sentence splitter | Control chunk boundaries at sentence level |

### Why BGE for POC

BGE-large produces a single 1024-dimensional vector per passage, which keeps the storage and retrieval pipeline straightforward: one vector per passage, standard ANN search, no multi-vector indexing complexity. This is sufficient to validate the core hypothesis -- that review passages embed near semantically matching queries -- without the operational overhead of ColBERT's multi-vector representations and PLAID indexing.

### ColBERT as Upgrade Path

ColBERT's late interaction mechanism computes similarity as a sum of maximum cosine similarities between each query token and all document tokens. This means when a user queries "stiff flex," the token "stiff" independently finds the most relevant passage token, rather than being averaged into a single vector that dilutes it. If BGE retrieval quality proves insufficient during POC evaluation, ColBERT is the natural next step.

```
score(Q, D) = sum over q_i in Q of max over d_j in D of (q_i . d_j)
```


## 3. Data Model

### Product Representation

Each product is stored as a **collection of passage vectors**, not a single vector. This is the multi-vector representation that makes the system work.

```python
@dataclass
class ProductDoc:
    product_id: str
    name: str
    domain: str                    # "ski", "running_shoe", "cookie"
    metadata: dict                 # price, brand, year -- stored as payload, not embedded
    passage_vectors: list[dict]    # [{"text": "...", "vector": [...], "source_review_id": "..."}]
```

### Passage Construction

Each review is split into **opinion-bearing passages** (1-3 sentences). Passages that are purely logistical ("shipped fast, good packaging") are filtered out via a lightweight classifier or keyword heuristic.

A product with 50 reviews might yield 120-200 passages, each independently embedded. This means the vector index contains ~150 vectors per product, not 1.

### Storage Layout (ChromaDB)

```
Collection: "product_passages"
  - embedding: float32[1024] (BGE single-vector)
  - metadata:
      product_id: str
      product_name: str
      passage_text: str
      review_id: str
      domain: str
  - document: passage text (for ChromaDB's built-in full-text support)
```


## 4. Ingestion Pipeline

```
Raw Reviews (JSON/CSV)
    |
    v
[1] Sentence segmentation (spaCy or regex)
    |
    v
[2] Opinion filtering (drop non-opinion sentences)
    |
    v
[3] Passage assembly (1-3 sentences, ~50-150 tokens)
    |
    v
[4] Embedding (BGE-large-en-v1.5)
    |
    v
[5] Upsert to ChromaDB with product_id metadata
```

### Step 2: Opinion Filtering

A simple heuristic: keep sentences containing adjectives, comparatives, or domain signal words. Drop sentences that are purely about shipping, customer service, or unboxing. This can be a 20-line rule-based filter for the POC.

```python
SKIP_PATTERNS = [
    r"\b(ship|deliver|packag|return|refund|box|arrived)\b",
    r"\b(customer service|support ticket)\b",
]

def is_opinion_bearing(sentence: str) -> bool:
    if any(re.search(p, sentence, re.I) for p in SKIP_PATTERNS):
        return False
    # keep if it has at least one adjective (crude but effective)
    doc = nlp(sentence)
    return any(tok.pos_ == "ADJ" for tok in doc)
```

### Step 3: Chunking Strategy

Passages are built by sliding a window of 1-3 sentences with 1-sentence overlap. This ensures no opinion spans a chunk boundary without appearing in at least one chunk.

### Batch Processing

For a catalog of 500 products with ~50 reviews each, expect ~75K passages. BGE-large encoding on a single GPU takes ~5 minutes. On CPU, ~1 hour. Acceptable for POC.


## 5. Query Pipeline

```
User Query: "stiff, on-piste carving ski, 180cm+"
    |
    v
[1] Query encoding (BGE-large, positive + contrastive negative)
    |
    v
[2] Retrieval from ChromaDB
    |        - retrieve top-200 passages
    |        - contrastive scoring (pos - neg similarity)
    |        - aggregate by product_id
    |
    v
[3] Product-level scoring
    |        - sum of top-K passage scores per product
    |        - normalize to 0-1 range
    |
    v
[4] Attribute extraction on evidence passages
    |        - regex/keyword matching for structured attributes
    |
    v
[5] Cross-encoder reranking (top-30 products)
    |        - concatenate query + top-3 passages per product
    |
    v
[6] Return top-10 products with evidence + matched attributes
```

### Passage-to-Product Aggregation

This is the critical step. Raw retrieval returns passages, but users want products. The aggregation incorporates contrastive scoring (Section 6) and normalizes to 0-1:

```python
def aggregate_passages(
    passage_texts: list[str],
    passage_product_ids: list[str],
    model,
    pos_query: str,
    neg_query: str,
    top_k_per_product: int = 5,
) -> list[dict]:
    """Score each product using contrastive passage scoring, normalized to 0-1."""
    import numpy as np

    # Encode positive and negative query vectors
    pos_vec = model.encode(pos_query, normalize_embeddings=True)
    neg_vec = model.encode(neg_query, normalize_embeddings=True)

    # Encode all passages
    passage_vecs = model.encode(passage_texts, normalize_embeddings=True)

    # Contrastive score per passage: pos_similarity - neg_similarity
    pos_scores = passage_vecs @ pos_vec  # cosine sim (already normalized)
    neg_scores = passage_vecs @ neg_vec
    contrastive_scores = pos_scores - neg_scores  # range: roughly [-2, 2]

    # Group by product
    product_scores = defaultdict(list)
    product_evidence = defaultdict(list)

    for i, pid in enumerate(passage_product_ids):
        product_scores[pid].append(float(contrastive_scores[i]))
        product_evidence[pid].append(passage_texts[i])

    # Sum-of-top-K aggregation
    results = []
    for pid, scores in product_scores.items():
        top = sorted(scores, reverse=True)[:top_k_per_product]
        results.append({
            "product_id": pid,
            "raw_score": sum(top),
            "evidence": product_evidence[pid][:3],
        })

    # Normalize scores to 0-1 (min-max across results)
    if results:
        raw_scores = [r["raw_score"] for r in results]
        min_s, max_s = min(raw_scores), max(raw_scores)
        score_range = max_s - min_s if max_s > min_s else 1.0
        for r in results:
            r["score"] = (r["raw_score"] - min_s) / score_range

    return sorted(results, key=lambda x: x["score"], reverse=True)
```

**Sum-of-top-K** (not mean) rewards products that match on multiple attributes mentioned in the query. A ski that has strong passages matching "stiff" AND "on-piste" AND "carving" will outscore one that only nails "stiff."

**Contrastive scoring** is the key differentiator. Instead of raw cosine similarity, each passage is scored by `sim(passage, pos_query) - sim(passage, neg_query)`. This creates a pseudo-ordinal ranking: a passage saying "incredibly stiff, no forgiveness" scores much higher than "medium flex" because the former is far from the negative query "soft forgiving ski."

**Score normalization** uses min-max scaling across the result set so all scores fall in [0, 1]. This produces bounded, comparable scores regardless of the number of matching passages or the magnitude of contrastive differentials.

### Attribute Extraction from Evidence

To produce structured `matched_attributes` for the common interface, a lightweight extraction step runs over the top evidence passages for each product:

```python
# Domain-agnostic attribute patterns. Additional domains add entries, not code.
ATTRIBUTE_PATTERNS = {
    "stiffness": r"\b(stiff|rigid|firm|soft|flexible|forgiving)\b",
    "weight": r"\b(light|lightweight|heavy|hefty|featherweight)\b",
    "stability": r"\b(stable|steady|planted|wobbly|unstable|locked.in)\b",
    "dampness": r"\b(damp|smooth|chattery|vibrat\w+|composed)\b",
    "speed": r"\b(fast|quick|slow|sluggish|responsive|snappy)\b",
    "grip": r"\b(grip|edge.hold|icy|slip|traction|bite)\b",
}

def extract_attributes(evidence: list[str], query: str, model) -> dict[str, float]:
    """
    Extract matched attributes from evidence passages.
    Returns attribute -> relevance score (0-1) based on keyword hits
    weighted by passage-query similarity.
    """
    import re
    query_vec = model.encode(query, normalize_embeddings=True)
    attr_scores: dict[str, list[float]] = defaultdict(list)

    for passage in evidence:
        p_vec = model.encode(passage, normalize_embeddings=True)
        sim = float(p_vec @ query_vec)  # 0-1 for normalized vecs

        for attr_name, pattern in ATTRIBUTE_PATTERNS.items():
            if re.search(pattern, passage, re.IGNORECASE):
                attr_scores[attr_name].append(sim)

    # Average similarity for each matched attribute, clamped to [0, 1]
    return {
        attr: min(1.0, max(0.0, sum(scores) / len(scores)))
        for attr, scores in attr_scores.items()
    }
```

This is deliberately simple -- regex keyword matching weighted by embedding similarity. It produces structured output (e.g., `{"stiffness": 0.82, "dampness": 0.71}`) without requiring a full NLP extraction pipeline. The patterns are domain-agnostic; adding a new domain means adding keyword entries to `ATTRIBUTE_PATTERNS`.

### Hard Constraint Handling

Numeric constraints like "180cm+" cannot be handled by embeddings alone. These are extracted with simple regex and applied as metadata filters in ChromaDB before retrieval.

```python
NUMERIC_PATTERNS = {
    "length_cm": r"(\d{2,3})\s*cm",
    "weight_g": r"(\d{2,4})\s*g\b",
    "price": r"\$(\d+)",
}
```

This is the one concession to structured data. It is minimal and domain-agnostic (just numbers + units).


## 6. Ranking Strategy: The Attribute Problem

The honest concern: **embeddings encode semantic similarity, not attribute intensity**. "Rock-solid stiff" and "medium stiff" will both match a query for "stiff ski" with similar scores, because embedding distance measures topic relevance, not degree.

### Mitigation 1: Contrastive Passage Scoring

Instead of asking "how similar is this passage to the query?", use a prompt-augmented approach. Embed both the positive query AND a negated version, then score by differential:

```python
pos_query = "stiff carving ski"
neg_query = "soft forgiving ski"

pos_score = similarity(passage, encode(pos_query))
neg_score = similarity(passage, encode(neg_query))
attribute_score = pos_score - neg_score  # high = truly stiff, near zero = ambiguous
```

This creates a pseudo-ordinal ranking. A passage saying "incredibly stiff, no forgiveness" will have a large differential. A passage saying "medium flex" will have a small one.

### Mitigation 2: Sentiment-Weighted Scoring

Use a lightweight sentiment model on the matched passage to detect intensity. Passages with stronger positive sentiment toward the queried attribute get boosted.

### Mitigation 3: Review Consensus

If 40 out of 50 reviewers mention "stiff" for product A, but only 5 out of 50 for product B, the passage count naturally boosts A. The sum-of-top-K aggregation already captures this: more matching passages = higher aggregate score.

### Honest Assessment

These mitigations help but do not fully solve the problem. For a query like "rank these skis from stiffest to softest," a pure embedding approach will underperform a system with explicit ordinal attribute extraction. The contrastive scoring gets us 70-80% of the way there; the remaining gap is real.


## 7. Domain Adaptation

This is where the pure embedding approach genuinely shines. Because there is no structured schema, adding a new domain requires:

1. Ingest reviews for the new domain.
2. Done.

No attribute ontology to define. No extraction rules to write. No domain-specific training. The embedding model already understands "crispy" for cookies, "responsive" for shoes, and "poppy" for skis, because it learned these in pretraining.

### Cross-Domain Query

A user can even make cross-domain analogies: "I want a ski that feels like a sports car, not a minivan." The embedding space captures this metaphorical mapping because the model has seen both automotive and ski reviews in its training data.

### Limitations

Truly niche domains with specialized jargon (e.g., "the beer has good lacing" or "excellent retrohale on this cigar") may not be well-represented in the embedding model's training data. Fine-tuning on domain reviews would help but adds operational cost.


## 8. Pros and Cons

### Pros

- **Zero schema design per domain.** Add a new product category by ingesting reviews. No ontology work.
- **Handles novel attributes automatically.** If reviewers start describing a new product characteristic, it is immediately searchable.
- **Natural language in, natural language out.** No query parsing, no attribute mapping, no structured query language.
- **Evidence-first results.** Every recommendation comes with the exact review passages that matched, providing built-in explainability.
- **Contrastive scoring adds ordinal signal.** The pos/neg differential scoring goes beyond vanilla semantic search to capture attribute intensity.
- **Simple stack.** BGE + ChromaDB keeps dependencies minimal with a clear upgrade path to ColBERT if needed.

### Cons

- **Ordinal ranking is weak.** Cannot reliably rank "stiffest to softest" without structured attributes. Mitigations help but do not fully solve this.
- **Storage cost.** Multi-vector representations (ColBERT upgrade path) use 50-100x more storage than single-vector. The BGE POC keeps this manageable. For 500K+ products at scale, storage optimization becomes a concern.
- **Embedding model quality ceiling.** The system is only as good as the embedding model's understanding of domain language. No way to inject expert knowledge about attribute relationships.
- **No attribute decomposition.** Cannot answer "how stiff is product X?" with a number. Only with evidence passages.
- **Cold start for sparse reviews.** Products with 1-2 short reviews have thin vector representations and will underperform in retrieval.
- **Latency.** Contrastive scoring requires two query embeddings and dual similarity computation. For POC scale this is fine; at 1M+ products it requires optimization (pre-filtering, caching, or upgrading to ColBERT's PLAID indexing).


## 9. POC Scope

### Goal

End-to-end demo: ingest ski reviews, query with natural language, return ranked products with evidence passages.

### Dataset

Scrape or manually collect ~200 reviews across ~20 ski models. Alternatively, use a public product review dataset and filter to a single category.

### Deliverables

1. Ingestion script that chunks reviews and embeds with BGE-large
2. ChromaDB index (in-process, ephemeral or persistent mode)
3. Query script implementing contrastive scoring, attribute extraction, and the common `Recommender` interface
4. Evaluation: manually judge top-5 results for 10 test queries

### Code Sketch: End-to-End

```python
# recommender.py
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Protocol
import json, uuid, re

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


# --- Common interface (shared across all designs) ---

@dataclass
class RecommendationResult:
    product_id: str
    product_name: str
    score: float           # 0-1 normalized
    explanation: str
    matched_attributes: dict[str, float]


class Recommender(Protocol):
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None: ...
    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]: ...


# --- Contrastive query generation ---

NEGATION_MAP = {
    "stiff": "soft forgiving", "soft": "stiff rigid", "light": "heavy",
    "heavy": "light", "fast": "slow sluggish", "slow": "fast quick",
    "stable": "unstable wobbly", "damp": "chattery vibrating",
    "responsive": "sluggish dead", "grippy": "slippery icy",
}

def generate_neg_query(query: str) -> str:
    """Generate a contrastive negative query by flipping key terms."""
    words = query.lower().split()
    neg_words = []
    for w in words:
        if w in NEGATION_MAP:
            neg_words.append(NEGATION_MAP[w])
        # skip non-attribute words (they stay the same to anchor domain context)
        else:
            neg_words.append(w)
    return " ".join(neg_words)


# --- Attribute extraction ---

ATTRIBUTE_PATTERNS = {
    "stiffness": r"\b(stiff|rigid|firm|soft|flexible|forgiving)\b",
    "weight": r"\b(light|lightweight|heavy|hefty|featherweight)\b",
    "stability": r"\b(stable|steady|planted|wobbly|unstable|locked.in)\b",
    "dampness": r"\b(damp|smooth|chattery|vibrat\w+|composed)\b",
    "speed": r"\b(fast|quick|slow|sluggish|responsive|snappy)\b",
    "grip": r"\b(grip|edge.hold|icy|slip|traction|bite)\b",
}

def extract_attributes(evidence: list[str], query_vec, model) -> dict[str, float]:
    """Extract matched attributes from evidence passages via keyword + similarity."""
    attr_scores: dict[str, list[float]] = defaultdict(list)
    for passage in evidence:
        p_vec = model.encode(passage, normalize_embeddings=True)
        sim = float(p_vec @ query_vec)
        for attr_name, pattern in ATTRIBUTE_PATTERNS.items():
            if re.search(pattern, passage, re.IGNORECASE):
                attr_scores[attr_name].append(sim)
    return {
        attr: round(min(1.0, max(0.0, sum(s) / len(s))), 3)
        for attr, s in attr_scores.items()
    }


# --- Implementation ---

class EmbeddingRecommender:
    """Pure embedding recommender with contrastive scoring."""

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.chroma = chromadb.Client()  # in-process, ephemeral
        self.product_names: dict[str, str] = {}

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        collection = self.chroma.get_or_create_collection(
            name=f"passages_{domain}",
            metadata={"hnsw:space": "cosine"},
        )
        # Index product names
        for p in products:
            self.product_names[p["product_id"]] = p.get("name", p["product_id"])

        # Chunk reviews into passages
        passages, metas, ids = [], [], []
        for review in reviews:
            chunks = self._chunk_review(review)
            for chunk in chunks:
                passages.append(chunk["text"])
                metas.append({
                    "product_id": chunk["product_id"],
                    "review_id": chunk["review_id"],
                    "domain": domain,
                })
                ids.append(str(uuid.uuid4()))

        # Embed and upsert in batches
        vectors = self.model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
        batch_size = 512
        for i in range(0, len(passages), batch_size):
            j = min(i + batch_size, len(passages))
            collection.add(
                ids=ids[i:j],
                embeddings=[v.tolist() for v in vectors[i:j]],
                documents=passages[i:j],
                metadatas=metas[i:j],
            )

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        collection = self.chroma.get_collection(f"passages_{domain}")

        # Encode positive and negative queries
        neg_query = generate_neg_query(query_text)
        pos_vec = self.model.encode(query_text, normalize_embeddings=True)
        neg_vec = self.model.encode(neg_query, normalize_embeddings=True)

        # Retrieve top-200 passages by positive query
        results = collection.query(
            query_embeddings=[pos_vec.tolist()],
            n_results=200,
            where={"domain": domain},
            include=["embeddings", "documents", "metadatas"],
        )

        docs = results["documents"][0]
        embeds = np.array(results["embeddings"][0])
        metas = results["metadatas"][0]

        # Contrastive scoring: pos_similarity - neg_similarity
        pos_scores = embeds @ pos_vec
        neg_scores = embeds @ neg_vec
        contrastive = pos_scores - neg_scores

        # Aggregate by product (sum-of-top-K)
        product_scores = defaultdict(list)
        product_evidence = defaultdict(list)
        for i, meta in enumerate(metas):
            pid = meta["product_id"]
            product_scores[pid].append(float(contrastive[i]))
            product_evidence[pid].append(docs[i])

        scored = []
        for pid, scores in product_scores.items():
            top = sorted(scores, reverse=True)[:5]
            scored.append({"product_id": pid, "raw": sum(top), "evidence": product_evidence[pid][:5]})

        # Normalize to 0-1
        if scored:
            raws = [s["raw"] for s in scored]
            min_s, max_s = min(raws), max(raws)
            rng = max_s - min_s if max_s > min_s else 1.0
            for s in scored:
                s["score"] = (s["raw"] - min_s) / rng

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Build results with attribute extraction
        output = []
        for item in scored[:top_k]:
            pid = item["product_id"]
            attrs = extract_attributes(item["evidence"][:3], pos_vec, self.model)
            output.append(RecommendationResult(
                product_id=pid,
                product_name=self.product_names.get(pid, pid),
                score=round(item["score"], 4),
                explanation=f"Matched on {len(attrs)} attributes from {len(item['evidence'])} review passages.",
                matched_attributes=attrs,
            ))
        return output

    def _chunk_review(self, review: dict) -> list[dict]:
        sentences = _segment_sentences(review["text"])
        chunks = []
        for i in range(0, len(sentences), 2):
            chunk = " ".join(sentences[i:i+3])
            if is_opinion_bearing(chunk):
                chunks.append({
                    "product_id": review["product_id"],
                    "text": chunk,
                    "review_id": review["review_id"],
                })
        return chunks


def _segment_sentences(text: str) -> list[str]:
    """Simple sentence splitter (replace with spaCy for production)."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


# Usage example
if __name__ == "__main__":
    rec = EmbeddingRecommender()

    # rec.ingest(products=[...], reviews=[...], domain="ski")
    # results = rec.query("stiff damp carving ski for hardpack", domain="ski")
    # for r in results:
    #     print(f"\n{r.product_name} (score: {r.score:.3f})")
    #     print(f"  Attributes: {r.matched_attributes}")
    #     print(f"  {r.explanation}")
```

### POC Timeline

| Day | Task |
|-----|------|
| 1 | Data collection, chunking pipeline, opinion filter |
| 2 | Embedding + ChromaDB indexing, basic query pipeline |
| 3 | Aggregation tuning, contrastive scoring experiment |
| 4 | 10-query evaluation, write-up of findings |

### Key Question the POC Must Answer

Can contrastive passage scoring (Section 6, Mitigation 1) produce meaningful ordinal ranking for attributes like stiffness, weight, and responsiveness? If yes, this design is viable as a primary system. If no, it is best positioned as a retrieval layer feeding into a more structured ranking system (hybrid with Design #1 or similar).


## 10. Common Interface

All designs implement the same protocol so they can be benchmarked interchangeably. This design's `EmbeddingRecommender` (see POC code sketch in Section 9) implements this interface.

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

### Interface Contract

- **`ingest()`**: Accepts raw product metadata and review dicts. Chunks reviews into opinion-bearing passages, embeds with BGE-large, and upserts into a ChromaDB collection scoped by `domain`. Idempotent -- re-ingesting the same domain replaces the previous index.
- **`query()`**: Accepts a natural language query string. Returns up to `top_k` results, each with:
  - `score`: 0-1 normalized via min-max scaling of contrastive sum-of-top-K scores.
  - `explanation`: Human-readable summary of why this product matched (evidence passage count and attribute match count).
  - `matched_attributes`: Dict of attribute name to relevance score (0-1), extracted via regex keyword matching on evidence passages weighted by embedding similarity. Example: `{"stiffness": 0.82, "dampness": 0.71, "stability": 0.65}`.

### Benchmark Compatibility

The `EmbeddingRecommender` class can be instantiated and used identically to any other design's recommender:

```python
from design_02 import EmbeddingRecommender

rec = EmbeddingRecommender()
rec.ingest(products=catalog, reviews=reviews, domain="ski")
results = rec.query("stiff damp carving ski for hardpack", domain="ski", top_k=10)

for r in results:
    assert 0.0 <= r.score <= 1.0
    assert isinstance(r.matched_attributes, dict)
```
