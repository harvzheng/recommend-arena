# Design #13: Researched-Distilled Hybrid

## 0. Status

New entry. Targets the deployable "recommend me an X" use case: user supplies a product catalog and a domain name, the offline pipeline produces two small artifacts, and the online recommender runs entirely on CPU with no API calls at query time.

This design supersedes neither #11 (fine-tuned MiniLM, current leader at NDCG@5=0.527) nor #12 (distilled Qwen 0.5B, NDCG@5=0.186). It composes the retrieval discipline of #11 with a corrected version of #12's distillation idea, then adds two things neither has: a *researched* teacher, and a generative explainer that runs only on the top-K.


## 1. Architecture Overview

The bet has three parts:

1. **A researched teacher beats a single-pass teacher.** #12's teacher (Claude/GPT-4o, one prompt per pair) produces shallow judgments. Real recommendation reasoning involves looking things up — what does "ice coast" mean, is this ski actually known for chatter, what do reviewers say about its weight relative to its category. A teacher that can run web search and read external reviews produces judgments that are anchored to evidence outside the local review corpus, which is small and noisy.
2. **Distill into a bi-encoder, not a generative ranker.** #12's structural mistake was making the student emit JSON per product. That puts an LLM forward pass in the inner loop of ranking, which is slow, brittle, and where the parse-failure rate (visible in #12's results) destroys precision. The right place to put the teacher's signal is in the embedding geometry — same as #11, but with supervised scores instead of template-derived contrastive pairs.
3. **Generate explanations only for the top-K.** Explanations are a UX deliverable, not a ranking signal. A 77M-parameter seq2seq model can produce a fluent explanation in ~15ms per item, batched across 10 items in ~150ms total. This is the same total budget as one Qwen-0.5B forward pass in #12, but produces explanations for the entire result set instead of just one product.

### Core flow

```
                       OFFLINE  (per domain, once)
                       ==========================

  Catalog + Reviews                        Domain name
        │                                       │
        ▼                                       ▼
  ┌───────────────────────────────────────────────────┐
  │  Phase 1: Synthetic Query Generation              │
  │  Opus (no research) writes 200-500 queries        │
  │  spanning easy / medium / hard / vague /          │
  │  cross-domain difficulty buckets                  │
  └───────────────────┬───────────────────────────────┘
                      │
                      ▼
  ┌───────────────────────────────────────────────────┐
  │  Phase 2: Researched Teacher Judgments            │
  │  Opus + web_search tool, one call per             │
  │  (query, product) pair                            │
  │  Output: {score, matched_attrs, explanation,      │
  │           evidence_quotes}                        │
  └───────┬───────────────────────────┬───────────────┘
          │                           │
          ▼                           ▼
  ┌──────────────────┐       ┌────────────────────┐
  │  Score-derived   │       │  (top-K, query,    │
  │  contrastive     │       │  explanation)      │
  │  triples         │       │  pairs, top-K only │
  └──────┬───────────┘       └─────────┬──────────┘
         │                             │
         ▼                             ▼
  ┌──────────────────┐       ┌────────────────────┐
  │ Bi-encoder train │       │ Explanation head   │
  │ MiniLM-L6-v2     │       │ flan-t5-small (77M)│
  │ MNR loss         │       │ supervised SFT on  │
  │ + margin loss on │       │ teacher rationales │
  │ score gradient   │       │ + matched_attrs    │
  └────────┬─────────┘       └─────────┬──────────┘
           │                           │
           ▼                           ▼
    retrieval.pt (~90MB)        explainer.onnx (~80MB int8)


                       ONLINE  (per query, local CPU)
                       ============================

  "stiff carving ski for hardpack"
              │
              ▼
       ┌────────────────┐
       │ MiniLM encoder │  ~5ms  (one forward pass)
       └──────┬─────────┘
              ▼
       ┌────────────────┐
       │ Cosine sim vs  │  <1ms  (matrix multiply)
       │ product matrix │
       └──────┬─────────┘
              ▼
        Top-K candidates (K=10 by default)
              │
              ▼
       ┌────────────────┐
       │ flan-t5-small  │  ~150ms total, batched
       │ over (query,   │  produces explanation +
       │ candidate ctx) │  matched_attributes per item
       └──────┬─────────┘
              ▼
        RecommendationResult[]
```

### Total query latency target

| Stage | Time (CPU, M-series laptop) |
|-------|----------------------------|
| Encode query | ~5ms |
| Cosine vs ≤500 products | <1ms |
| Explain top-10 (batched) | ~150ms |
| **Total p50** | **~160ms** |

For comparison: #11 is ~6ms but has no NL explanations. #12 is ~5-15s and frequently produces malformed JSON. #3 (LLM-judge) is 10-25s with API calls.

### Why this might not work

- **Researched teacher is expensive.** Opus + web_search on every (query, product) pair runs 4-10× the token cost of the #12 teacher and 5-30× the wallclock time. For 500 pairs this is dollars, not pennies. Acceptable as a one-time cost; not acceptable if we have to relabel often.
- **Score signal may be too dense.** #11 uses binary positives via `MultipleNegativesRankingLoss`. A continuous teacher score (0.0-1.0) is richer but harder to fit. Translation to contrastive pairs requires a thresholding policy that can lose information.
- **Explainer drift.** flan-t5-small is small enough to overfit teacher phrasing. If the teacher always uses the word "damp" near the word "composed," the student will too — even when the underlying product evidence does not support that pairing. Mitigation in §7.


## 2. Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Teacher LLM** | Claude Opus (latest) via Anthropic SDK | Highest reasoning quality available; native tool use for `web_search` |
| **Teacher tools** | `web_search`, `web_fetch` | Lets teacher cross-reference reviewer claims against external sources |
| **Retrieval base model** | `all-MiniLM-L6-v2` (22M params, 384-dim) | Same as #11. Proven at this scale. CPU-friendly. |
| **Retrieval training** | `sentence-transformers` + custom loss | MNR loss + margin loss with continuous score targets |
| **Explainer base model** | `google/flan-t5-small` (77M params) | Pre-trained for instruction-following seq2seq; small enough for CPU; reliable structured output via prompt templates |
| **Explainer training** | HuggingFace `Trainer` + LoRA (rank 8) | LoRA keeps adapter ~3MB, base model can be shared/cached |
| **Explainer inference** | `transformers` + ONNX Runtime int8 export | Cuts CPU latency 2-3x vs vanilla PyTorch |
| **Vector store** | numpy in-memory | <500 products per domain; matrix multiply is faster than any ANN index |
| **Catalog store** | SQLite (one file per domain) | Same convention as #04, #05, #09 |
| **Serving** | FastAPI + uvicorn | Standard, async, OpenAPI for free |
| **Container** | python:3.11-slim + uv | <600MB image with both model files baked in |
| **Teacher cache** | SQLite | Required for resumability — researched calls take minutes each |

### Why MiniLM over BGE for retrieval

Same reasons as #11: 47x smaller, 5x faster on CPU, less overfitting risk on small datasets. The empirical result from #11 (0.527 NDCG@5) confirms this is the right base model.

### Why flan-t5-small over a decoder-only model

The explainer task is short-input → short-output, structured generation. flan-t5-small is purpose-built for instruction-tuned conditional generation in this regime. Decoder-only models in this size class (TinyLlama, Qwen-0.5B as in #12) are biased toward fluent free-form generation and have to be fought into producing structured output. T5's encoder-decoder split also means the encoder can be shared across all top-K items in a query (one pass per product), with the decoder running the per-item generation — a meaningful efficiency gain at K=10.

### Why not Gemma-2B / larger student

Gemma-2B is 2.5GB on disk, ~4GB resident at fp16, ~1GB at int4. That is too large for the "deploy anywhere" constraint and we cannot demonstrate it adds value over the bi-encoder + small explainer split. If the explainer proves insufficient, upgrading to flan-t5-base (250M) is the next step before considering Gemma.


## 3. Data Model

### 3.1 Teacher-side artifacts

```python
from dataclasses import dataclass

@dataclass
class SyntheticQuery:
    query_id: str          # "syn-001"
    text: str              # "freeride ski with surfy feel and good float"
    difficulty: str        # "easy" | "medium" | "hard" | "vague"
    seed_attributes: list[str]  # attributes the generator was prompted to target
    domain: str

@dataclass
class EvidenceQuote:
    """A passage the teacher cited as evidence for its judgment."""
    source: str            # "review_id:r042" or "https://blistergearreview.com/..."
    text: str              # the quoted passage
    relevance: float       # teacher's self-rated relevance, 0-1

@dataclass
class TeacherJudgment:
    query_id: str
    product_id: str
    score: float                      # 0.0 - 1.0
    matched_attributes: dict[str, float]  # attribute -> match strength
    explanation: str                  # 2-4 sentence rationale
    evidence: list[EvidenceQuote]     # what the teacher used to decide
    teacher_model: str                # "claude-opus-4-7"
    research_calls: int               # how many tool calls the teacher made
    created_at: str                   # ISO-8601 UTC
```

### 3.2 Retrieval-side artifacts

```python
import numpy as np

@dataclass
class TrainingTriple:
    """Score-derived contrastive triple for retrieval training."""
    query: str
    positive_passage: str       # passage from a product the teacher scored high
    negative_passage: str       # passage from a product the teacher scored low
    score_margin: float         # teacher_score(pos) - teacher_score(neg), used as margin

@dataclass
class ProductVector:
    product_id: str
    product_name: str
    vector: np.ndarray          # 384-dim, L2-normalized, mean of passage embeddings
    passage_texts: list[str]    # kept for explainer context
    metadata: dict              # raw product fields (price, brand, specs)
```

### 3.3 Explainer-side artifacts

```python
@dataclass
class ExplainerExample:
    """One SFT example for the explainer head."""
    input: str    # rendered prompt: query + product context
    output: str   # rendered target: matched_attrs JSON + explanation prose

@dataclass
class ExplanationRequest:
    """One inference call to the explainer."""
    query: str
    product: ProductVector
    rank: int                # position in retrieval ranking, 1-indexed
    raw_score: float         # cosine similarity, for explainer context
```

### 3.4 SQLite schema (per domain)

```sql
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    review_text TEXT NOT NULL,
    source TEXT
);

CREATE TABLE synthetic_queries (
    query_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    seed_attributes_json TEXT NOT NULL,
    domain TEXT NOT NULL
);

CREATE TABLE teacher_judgments (
    query_id TEXT NOT NULL REFERENCES synthetic_queries(query_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    score REAL NOT NULL,
    matched_attributes_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    teacher_model TEXT NOT NULL,
    research_calls INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (query_id, product_id, teacher_model)
);

CREATE INDEX idx_judgments_query ON teacher_judgments(query_id);
CREATE INDEX idx_judgments_score ON teacher_judgments(score);
```


## 4. Phase 1 — Synthetic Query Generation

The benchmark dataset has 20 hand-curated queries. That is enough to *evaluate* but nowhere near enough to *train*. Phase 1 generates a much larger, deliberately diverse set of queries against which the teacher will produce judgments.

### 4.1 Generation strategy

For a given domain (say, ski), enumerate the structured attribute space (stiffness, edge_grip, dampness, …) plus metadata fields (price, weight, brand). Bucket combinations into the same five difficulty buckets the benchmark uses. Prompt Opus with one bucket and one attribute combination per call:

```
You are generating product search queries that real users would type.
Domain: {domain}
Difficulty bucket: {bucket}      # easy | medium | hard | vague | cross_domain
Attribute focus: {attributes}    # ["stiffness:high", "edge_grip:high"]

Easy queries: single clear attribute, plain language.
Medium queries: 2-3 constraints, may mix attributes and metadata.
Hard queries: include negations, ranges, or trade-offs.
Vague queries: subjective or metaphorical, no explicit attribute names.

Generate 5 distinct queries for this bucket and attribute focus.
Avoid templated phrasing. Vary length, tone, and word choice.

Return JSON: {"queries": [{"text": "...", "rationale": "..."}]}
```

### 4.2 Volume and balance

| Bucket | Queries per attribute combo | Combos | Per domain total |
|--------|----------------------------|--------|------------------|
| Easy | 5 | 8 single-attribute | 40 |
| Medium | 5 | 16 two-attribute | 80 |
| Hard | 5 | 12 attribute + metadata | 60 |
| Vague | 5 | 10 thematic | 50 |
| Cross-domain | 3 | 5 hand-picked | 15 |
| **Total** | | | **~245** |

This is comfortably enough training signal once paired with the catalog (~25 products → ~6,000 (q, p) pairs) without exploding teacher cost.

### 4.3 Disjoint from benchmark queries

The benchmark queries (defined in `benchmark/data/queries.json`) MUST NOT appear in the synthetic set. The pipeline will hash benchmark query text and reject any synthetic query whose normalized form (lowercased, whitespace-collapsed, trailing-punct-stripped) collides. This keeps the held-out set actually held out.


## 5. Phase 2 — Researched Teacher Judgments

This is where the design earns its name. The teacher does not just score from the local context; it researches.

### 5.1 Teacher prompt and tool calls

```
You are an expert {domain} product evaluator. You will judge how well a
product matches a user's query, and you have access to web search.

Query: {query_text}

Product: {product_name}
Local context (specs + reviews from our catalog):
{product_context}

Your job:
1. Decide whether you have enough information to judge confidently.
   If not, use web_search to look up authoritative reviews of this product.
   Cap research at 2 search calls + 1 fetch.
2. Resolve any jargon in the query (e.g., "ice coast" = East Coast hardpack).
   Use search if you are uncertain.
3. Score the match from 0.0 to 1.0 using this rubric:
   0.9-1.0: Near-perfect match on every attribute the query asks for.
   0.7-0.89: Strong match with minor gaps.
   0.5-0.69: Decent match but notable trade-offs.
   0.3-0.49: Partial match, significant misalignment.
   0.0-0.29: Poor match or wrong category.
4. Identify which attributes you matched on and the strength of each.
5. Quote the specific evidence (from local context or web sources) you used.
6. Write a 2-4 sentence explanation that a user would find informative.

Output JSON only:
{
  "score": float,
  "matched_attributes": {"attribute_name": float, ...},
  "explanation": str,
  "evidence": [{"source": str, "text": str, "relevance": float}, ...]
}
```

Tool budget per (query, product) pair: max 3 calls (2 search + 1 fetch). The pipeline will hard-cap this on the SDK side using `max_tokens` and a tool-call counter to bound cost.

### 5.2 Caching

Every judgment goes into `teacher_judgments` keyed by `(query_id, product_id, teacher_model)`. Resuming a partially-completed labeling run skips already-cached pairs. This is the same pattern as #12 §4.3 and is essential — a researched run is minutes per pair, hours total.

### 5.3 Cost and time

| Pairs | Avg tokens (in+out) | Tool calls | Wall time | Estimated cost |
|-------|--------------------|-----------|-----------|---------------|
| 6,000 (245 queries × 25 products) | ~6K per pair | ~1.5 avg | 8-15 hours | $50-150 |

This is roughly two orders of magnitude more expensive than #12's teacher pass. It is acceptable because (a) it runs once per domain, (b) result quality is the headline differentiator, (c) caching makes incremental updates cheap.

### 5.4 Sanity checks

After labeling completes, the pipeline runs three checks before training:
- **Score distribution**: scores should not collapse to {0, 1}. If >80% of scores are at the extremes, the teacher rubric is broken.
- **Per-product variance**: each product should have at least one query where it scores high. If a product is uniformly low, that is a pipeline bug or a catalog data issue, not a model problem.
- **Tool-call usage**: median calls per pair should be 0.5-1.5. If it is 0, the teacher never researched (bad). If it is at the cap, the teacher always hit the limit (rubric too aggressive).


## 6. Phase 3 — Bi-encoder Training

This is structurally close to #11 §4 with the key change: pairs are derived from teacher *scores*, not from `attribute_score >= threshold`. That delta is where the design's accuracy lift is supposed to come from.

### 6.1 Triple construction

For each query, take its `TeacherJudgment` rows sorted by score. Form triples `(query, positive_passage, negative_passage)` where:
- positive comes from a product scoring ≥ 0.7
- negative comes from a product scoring ≤ 0.3
- score margin (`pos_score - neg_score`) is recorded for use as a margin-loss target

If the score distribution does not yield enough triples at those thresholds for a given query (e.g., a vague query where no product scored above 0.7), fall back to relative thresholds: top-quartile vs bottom-quartile of scores for that query.

```python
def build_triples(
    judgments_by_query: dict[str, list[TeacherJudgment]],
    products_by_id: dict[str, ProductProfile],
    pos_threshold: float = 0.7,
    neg_threshold: float = 0.3,
) -> list[TrainingTriple]:
    triples = []
    for query_id, js in judgments_by_query.items():
        query_text = SYNTHETIC_QUERIES[query_id].text
        scores = sorted(js, key=lambda j: j.score)
        positives = [j for j in scores if j.score >= pos_threshold]
        negatives = [j for j in scores if j.score <= neg_threshold]

        # Fallback to relative thresholds if absolute thresholds are sparse
        if len(positives) < 2 or len(negatives) < 2:
            n = len(scores)
            if n < 4:
                continue  # not enough signal for this query
            negatives = scores[: n // 4]
            positives = scores[-(n // 4) :]

        for pos in positives:
            pos_passages = products_by_id[pos.product_id].review_passages
            if not pos_passages:
                continue
            for neg in negatives:
                neg_passages = products_by_id[neg.product_id].review_passages
                if not neg_passages:
                    continue
                triples.append(TrainingTriple(
                    query=query_text,
                    positive_passage=random.choice(pos_passages),
                    negative_passage=random.choice(neg_passages),
                    score_margin=pos.score - neg.score,
                ))
    return triples
```

### 6.2 Loss function

Use `MultipleNegativesRankingLoss` (same as #11) with one addition: a margin-aware reweighting term that scales each triple's contribution by its `score_margin`. Triples where the teacher said "this is great vs this is terrible" (margin ~0.7) should weigh more than triples where the teacher said "this is decent vs this is okay" (margin ~0.4).

```python
import torch
from sentence_transformers.losses import MultipleNegativesRankingLoss

class MarginWeightedMNRLoss(MultipleNegativesRankingLoss):
    """MNR loss with per-example weight = sigmoid(margin * temperature).

    Triples with larger score margins (teacher confident the positive is much
    better than the negative) contribute more to the gradient. Triples with
    small margins (teacher saw a close call) contribute less, since they're
    weaker training signal.
    """
    def __init__(self, model, scale: float = 20.0, temperature: float = 4.0):
        super().__init__(model, scale=scale)
        self.temperature = temperature

    def forward(self, sentence_features, labels, margins=None):
        loss_per_example = super().forward(sentence_features, labels)
        if margins is None:
            return loss_per_example.mean()
        weights = torch.sigmoid(margins * self.temperature)
        return (loss_per_example * weights).mean()
```

### 6.3 Training config

```python
@dataclass
class RetrievalTrainingConfig:
    base_model: str = "all-MiniLM-L6-v2"
    epochs: int = 6
    batch_size: int = 32        # larger than #11 since we have more data
    learning_rate: float = 2e-5
    warmup_fraction: float = 0.10
    validation_split: float = 0.15
    margin_temperature: float = 4.0
    output_dir: str = "checkpoints/design-13/retrieval"
    seed: int = 42
```

Estimated training time at 6,000 pairs × 4-6 triples each = ~25K triples, batch 32, 6 epochs: ~10-15 minutes on CPU, ~1-2 minutes on a single GPU.

### 6.4 Eval during training

The validation set is a 15% holdout of synthetic queries (not the benchmark queries). For each held-out query, compute per-query NDCG@5 against teacher scores treated as relevance grades (rounded to {0,1,2,3} via thresholds 0.3/0.6/0.85). This is a proxy for benchmark NDCG and lets us early-stop without leaking test data.


## 7. Phase 4 — Explainer Training

The explainer is a small seq2seq model that, given (query, product context), produces a structured explanation matching the teacher's format.

### 7.1 SFT data construction

Only use teacher judgments where the *teacher* gave that product a high rank for that query. Specifically: per query, take the top-5 products by teacher score, and use those (q, p) pairs as positive explainer training examples. Reasoning: the explainer will only ever be invoked on top-K retrieval results at inference time, so training it on full positives/negatives wastes capacity on a regime it will never see.

```python
def build_explainer_examples(
    judgments_by_query: dict[str, list[TeacherJudgment]],
    products_by_id: dict[str, ProductProfile],
    top_per_query: int = 5,
) -> list[ExplainerExample]:
    examples = []
    for query_id, js in judgments_by_query.items():
        query_text = SYNTHETIC_QUERIES[query_id].text
        top = sorted(js, key=lambda j: -j.score)[:top_per_query]
        for j in top:
            ctx = render_product_context(products_by_id[j.product_id])
            input_text = (
                f"Query: {query_text}\n\n"
                f"Product: {j.product_name}\n"
                f"{ctx}\n\n"
                f"Rank: {top.index(j) + 1}/{top_per_query}"
            )
            output_text = json.dumps({
                "matched_attributes": j.matched_attributes,
                "explanation": j.explanation,
            })
            examples.append(ExplainerExample(input=input_text, output=output_text))
    return examples
```

### 7.2 LoRA-tuned flan-t5-small

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

EXPLAINER_LORA = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "k", "v", "o"],   # T5's attention projections
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

@dataclass
class ExplainerTrainingConfig:
    base_model: str = "google/flan-t5-small"
    epochs: int = 4
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_input_length: int = 1024
    max_output_length: int = 256
    output_dir: str = "checkpoints/design-13/explainer"
```

### 7.3 Drift mitigation

The risk noted in §1 is that the explainer parrots teacher phrasing instead of grounding in product evidence. Two mitigations:

1. **Evidence anchoring in the prompt template.** Include 2-3 evidence quotes from the local product context as part of the input. The explainer is conditioned on raw evidence, not just product specs, and is more likely to echo that evidence than confabulate.
2. **Held-out attribute filter.** Hold out 1-2 attributes from training (e.g., never train on judgments where the query targets `playfulness`). At eval, run the explainer on those held-out attributes and check that it can still produce reasonable explanations grounded in evidence — not just memorized teacher phrases.


## 8. Ingestion + Index Build

Identical in shape to #11 §5: chunk reviews into passages, encode each passage with the trained retrieval model, mean-pool per product, L2-normalize, save. The catalog and review tables go to SQLite. The product matrix is a single `product_vectors.npy` file.

The only meaningful difference from #11 is that ingestion can run in two modes:

- **Train-then-ingest (cold start)**: no checkpoint exists, so the pipeline runs phases 1-4 first, then ingests. This is the multi-hour path.
- **Ingest-only (warm)**: checkpoints exist and are compatible with the catalog (same domain, same attribute schema). Skip training, build the index in seconds.

The compatibility check hashes the catalog's attribute schema and compares it against the schema stamped into the checkpoint metadata. Mismatch forces retraining.


## 9. Query / Ranking Pipeline

```python
class ResearchedDistilledRecommender:
    """Design #13: researched-distilled hybrid.

    Two-stage at query time:
      1. Bi-encoder retrieval over full catalog (~5ms)
      2. Explainer over top-K only (~150ms batched)
    """

    def __init__(
        self,
        retrieval_dir: str = "checkpoints/design-13/retrieval",
        explainer_dir: str = "checkpoints/design-13/explainer",
        domain_db: str = "design_13.db",
    ):
        self.retrieval = SentenceTransformer(retrieval_dir)
        self.explainer_tok = AutoTokenizer.from_pretrained(explainer_dir)
        self.explainer = AutoModelForSeq2SeqLM.from_pretrained(explainer_dir)
        self.db = sqlite3.connect(domain_db)
        self._init_db()
        self.product_vectors: dict[str, list[ProductVector]] = {}

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        # See §8. Stores raw catalog, builds product vectors, persists.
        ...

    def query(
        self, query_text: str, domain: str, top_k: int = 10
    ) -> list[RecommendationResult]:
        if domain not in self.product_vectors:
            return []

        # Stage 1: retrieval
        qvec = self.retrieval.encode(query_text, normalize_embeddings=True)
        pvecs = self.product_vectors[domain]
        product_matrix = np.stack([pv.vector for pv in pvecs])
        scores = product_matrix @ qvec
        top_indices = np.argsort(-scores)[:top_k]
        candidates = [(pvecs[i], float(scores[i])) for i in top_indices]

        # Stage 2: explanation (batched)
        prompts = [self._build_prompt(query_text, pv, rank=i + 1, score=s)
                   for i, (pv, s) in enumerate(candidates)]
        inputs = self.explainer_tok(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        )
        with torch.no_grad():
            outputs = self.explainer.generate(
                **inputs, max_new_tokens=256, num_beams=1, do_sample=False,
            )
        rendered = self.explainer_tok.batch_decode(
            outputs, skip_special_tokens=True,
        )

        # Parse + assemble results
        results = []
        for (pv, raw_score), rendered_str in zip(candidates, rendered):
            parsed = self._parse_explainer_output(rendered_str)
            results.append(RecommendationResult(
                product_id=pv.product_id,
                product_name=pv.product_name,
                score=self._normalize(raw_score, scores),
                explanation=parsed.get("explanation", ""),
                matched_attributes=parsed.get("matched_attributes", {}),
            ))
        return results
```

### 9.1 Explainer parse robustness

The explainer is small and may produce malformed JSON occasionally (#12 §7 documents this issue). Mitigation:

- **Strict prompt template** with worked examples in the system message during training.
- **Greedy bracket extraction** (#12 §7.2) as fallback parser.
- **Final fallback**: if both parsers fail, return the raw text as the explanation and use the retrieval-stage `matched_attributes` derived from centroid decomposition (the #11 §7 mechanism), so the result is still well-formed for downstream consumers.

The retrieval-stage centroid decomposition is computed once at index build time and stored alongside `product_vectors.npy`; the cost at query time is one extra dot product per attribute, negligible.

### 9.2 No reranker

Same reasoning as #11 §6.2 — the bi-encoder is trained on supervised teacher scores, so the retrieval IS the ranking. The explainer does not change the order; it only annotates. If a future eval shows the explainer can produce useful re-ranking signals (e.g., flagging "wrong category" mistakes the bi-encoder makes), we can add a `score_adjustment` field to its output and apply it as a tiebreaker — but that is post-launch.


## 10. Serving Layer

The deployable product is a CLI + a FastAPI service.

### 10.1 CLI

```bash
# Train a recommender for a new domain end-to-end
recommend-arena train \
  --catalog catalog.jsonl \
  --reviews reviews.jsonl \
  --domain coffee \
  --teacher anthropic/claude-opus-4-7 \
  --output models/coffee/

# Serve a trained recommender
recommend-arena serve \
  --model-dir models/coffee/ \
  --port 8080

# Ad-hoc query (useful for debugging)
recommend-arena query \
  --model-dir models/coffee/ \
  --text "fruity natural-process beans for pour-over"
```

### 10.2 HTTP API

```
POST /recommend
{
  "query": "stiff carving ski for hardpack",
  "domain": "ski",          // optional, defaults to model's primary domain
  "top_k": 10
}
→ 200 OK
{
  "results": [
    {
      "product_id": "...",
      "product_name": "...",
      "score": 0.87,
      "explanation": "...",
      "matched_attributes": {"stiffness": 0.91, "edge_grip": 0.85}
    }, ...
  ],
  "latency_ms": 162,
  "model_version": "design-13-r1"
}

GET /health           → 200 if both checkpoints loaded, 503 otherwise
GET /domains          → list of domains the loaded model supports
```

### 10.3 Container

```dockerfile
FROM python:3.11-slim
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen
COPY src/ ./src/
COPY models/ ./models/                 # baked in
EXPOSE 8080
CMD ["uvicorn", "recommend_arena.serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

Image size target: <600MB (base ~150MB + deps ~200MB + retrieval 90MB + explainer 80MB + catalog SQLite ~20MB).


## 11. Benchmark Integration

The recommender satisfies the existing `Recommender` protocol unchanged. The training pipeline runs out-of-band:

```python
from implementations.design_13_researched_distilled import ResearchedDistilledRecommender

rec = ResearchedDistilledRecommender(
    retrieval_dir="checkpoints/design-13/retrieval",
    explainer_dir="checkpoints/design-13/explainer",
)
rec.ingest(products=catalog, reviews=reviews, domain="ski")
results = rec.query("stiff damp carving ski for hardpack", domain="ski", top_k=10)
```

### 11.1 Benchmark guardrails

Two guardrails specific to this design:

- **Held-out queries must remain held out.** The training pipeline asserts that no synthetic query collides with `benchmark/data/queries.json` (§4.3). The benchmark runner prints a warning if any benchmark query text appears verbatim in `synthetic_queries`.
- **Teacher cache must not be reused across designs.** This design's teacher cache is namespaced under `design_13_*` tables. Other designs must not read from it (they wouldn't anyway, but the namespace prevents accidents).

### 11.2 Expected results

Predicting empirically is hard, but the structural argument is:

- vs **#11**: same retrieval architecture, supervised by richer signal → expect 0.05-0.10 NDCG@5 lift, concentrated on `vague` and `hard` buckets where #11's templated training is weakest.
- vs **#12**: completely different scoring path → expect at minimum 0.20+ NDCG@5 lift, and ~50× faster query time.

If neither of those hold, the design has failed and we have learned something useful about whether researched-teacher signal actually transfers through distillation.


## 12. Trade-offs

### Strengths
- **Fully local at query time.** No API calls, no network, ~160ms p50 on CPU.
- **One-shot deployable.** Two model files, one SQLite, one container. No vector DB, no Ollama.
- **Natural-language explanations** that the leading retrieval design (#11) cannot produce.
- **Domain-agnostic ingestion.** User supplies catalog + reviews + domain name; pipeline does the rest.
- **Composes proven pieces.** Retrieval is structurally #11 (which works). Explainer is a small, well-understood seq2seq head. The new piece is the teacher and the score-derived training signal.

### Weaknesses
- **Long, expensive offline training.** 8-15 hours of teacher-side wall time, $50-150 in API spend per domain. First deployment is not "five minutes after I provided the catalog."
- **Two checkpoints to keep in sync.** A retrieval checkpoint and an explainer checkpoint trained from the same teacher cache. Versioning has to track both.
- **Explainer drift risk.** flan-t5-small can overfit teacher phrasing. §7.3 has mitigations, but the risk is real and only validated by running the held-out-attribute eval.
- **No real-time learning.** Same as #11 and #12 — once deployed, the model is frozen. User feedback would require re-running phase 2 on new queries.
- **Teacher quality ceiling.** Same fundamental limit as #12, mitigated but not removed by giving the teacher tools.

### Honest comparison

| Dimension | #11 (Fine-Tuned Embed) | #12 (Distilled LLM) | #13 (this) |
|-----------|------------------------|---------------------|------------|
| Retrieval supervision | Template-derived | (none, generative) | Researched teacher |
| Query latency (CPU) | ~6ms | 5-15s | ~160ms |
| NL explanations | No (centroid decomp) | Yes (often broken) | Yes (small explainer) |
| Per-query API cost | $0 | $0 | $0 |
| Deploy footprint | 90MB | 500-1500MB | 170MB |
| Offline training cost | $0 | ~$1 | $50-150 |
| Training time | 2-5 min | 5-15 min | 8-15 hours teacher + 15 min train |
| Domain transfer | Re-train | Re-train | Re-train (faster than full re-distill) |

## 13. Future Directions

- **Cached research per domain.** The teacher's web research output for "ice coast = East Coast hardpack" applies to *every* (query, product) pair touching that concept. Caching research at the term level rather than the (q, p) level could cut teacher cost 3-5×.
- **Cross-domain explainer.** Train the explainer on multiple domains simultaneously. flan-t5-small has plenty of capacity; sharing across domains may improve generalization to new domains with sparse training data.
- **DPO from user feedback.** Once deployed, "user clicked product X for query Y but skipped product Z" is a preference triple. DPO on those triples gives online adaptation without re-running phase 2.
- **GGUF + Ollama serving for the explainer.** llama.cpp now supports T5 architectures (encoder-decoder GGUF). Migrating the explainer to GGUF int4 would cut its disk footprint to ~30MB and CPU inference latency another ~2x.
- **Active labeling.** Instead of labeling all 6,000 (q, p) pairs, use uncertainty sampling: train the bi-encoder on a small initial label set, predict scores for unlabeled pairs, send only the high-uncertainty ones to the teacher. This could cut teacher cost in half with minimal quality loss.
