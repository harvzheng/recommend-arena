# recommend-arena

Fourteen recommendation-system designs, one shared interface, one shared dataset, one benchmark. Same queries go in; we measure which approach actually retrieves the right products and why.

The premise: instead of picking a recommender architecture upfront, build the major contenders, hold them to the same evaluation, and let NDCG decide. The original arena ran twelve designs against a 20-query ski set. The shipping configuration that came out of it is **design 14** — a local-first hybrid with a Rust hot path and per-domain bundles — evaluated against a 100-query v2 ski set and a 30-query wine set.

![NDCG@5 by design × query difficulty](docs/thumbnail.png)

## What's in here

```
designs/                # 14 design documents (the "spec" for each implementation)
implementations/        # 14 working Python packages — one per design
shared/                 # Recommender protocol, bundle format, LLM provider abstraction
arena_core/             # Rust hot path (PyO3): prefilter SQL, FTS5 retrieval, RRF fusion
scripts/                # Bundle compiler (arena_new.py), trainers, eval gate
benchmark/              # Test data, runner, ground-truth queries
benchmark/results/      # Historical arena evidence (designs 1–12, ski_eval v1)
artifacts/<domain>/     # Compiled bundles — manifest, FTS5 db, embeddings, eval matrix
frontend/               # Streamlit UI over any built bundle
docs/thumbnail.png      # The heatmap above (regenerable via scripts/build_heatmap.py)
```

Every implementation conforms to a single protocol so the runner can swap them in interchangeably:

```python
class Recommender(Protocol):
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None: ...
    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]: ...
```

## The fourteen designs

| # | Design | Approach |
|---|---|---|
| 01 | Knowledge Graph | NetworkX graph over products, attributes, terrain tags; query rewrites to graph traversal |
| 02 | Pure Embedding | ChromaDB + BGE embeddings over product+review text |
| 03 | LLM-as-Judge | RAG retrieval, then pointwise LLM scoring |
| 04 | Hybrid (SQL + Vector) | Dual-track SQLite filtering and vector recall, fused at the end |
| 05 | SQL + FTS5 | SQLite FTS5 with BM25 — pure lexical baseline |
| 06 | Multi-Agent Pipeline | Function pipeline (parse → retrieve → re-rank → explain) |
| 07 | Bayesian | Conjugate priors / Dirichlet-Multinomial over attribute evidence |
| 08 | TF-IDF | Sparse feature vectors with learned attribute weights |
| 09 | Faceted Search | SQLite-backed facet matcher, no semantic layer |
| 10 | Ensemble / LTR | XGBoost ranker over BM25 + FAISS + structured features |
| 11 | Fine-tuned Embeddings | Contrastive fine-tune of a small embedding model on synthetic query/product pairs |
| 12 | Distilled LLM | Small student model distilled from an LLM teacher |
| 13 | Frontier Listwise (Opus 4.7) | Full catalog stuffed into one prompt with cache-control; upper-bound reference |
| 14 | Local-First Hybrid | **Shipping design.** Rust hot path (prefilter + FTS5 + RRF) + Python embeddings + optional listwise rerank |

## The dataset

Two product domains let us check that nothing has overfit to skis:

- **Skis** — 25 products differentiated across race / carving / all-mountain / freeride / powder / freestyle / beginner; 132 reviews written to cover the attribute space (with deliberate overlap of phrasing between reviews and queries to stress semantic matching).
- **Running shoes** — 10 products, 50 reviews. Used by the cross-domain queries to verify each design can switch domains without code changes.

Two eval sets exist for skis. The original 12 designs were benchmarked on **v1** (20 queries); the local-hybrid design 14 results below use **v2** (100 queries, in `benchmark/data/per_domain/ski_eval_v2.json`). v2 keeps the same five buckets but at 5× the coverage:

| Bucket | v1 count | v2 count | What it tests |
|---|---|---|---|
| Easy | 5 | 25 | Single clear attribute (`"powder ski with good float"`) |
| Medium | 5 | 30 | Multiple constraints (`"titanal construction ski with edge grip for hardpack and at least 95mm waist"`) |
| Hard | 5 | 25 | Negations, ranges, trade-offs (`"freeride ski that is NOT playful, stiff with high stability, over 105mm waist"`) |
| Vague | 3 | 20 | Subjective / metaphorical (`"a ski that feels alive underfoot"`, `"good ski for the ice coast"`) |
| Cross-domain | 2 | — | Domain switch into running shoes (v1 only) |

A third domain landed: **wine** — a 200-product subsample of the HF wine-reviews dataset with a 30-query eval (`benchmark/data/per_domain/wine_eval.json`) graded by `openrouter/owl-alpha`. It exists to stress the framework on a domain very different from skis (price/points-driven, larger catalog, less structured vocabulary).

Each query carries a hand-curated `ground_truth_top5` with relevance grades (1–3), enabling standard NDCG.

## Metrics

- **NDCG@5 / @10** — primary ranking quality (relevance-weighted, position-discounted)
- **MRR** — how soon the first highly-relevant result appears
- **Attribute F1** — precision/recall of the attributes each design claims it matched on
- **Coverage** — fraction of queries that returned at least one relevant result
- **Explanation Quality** — automated proxy: does the explanation reference real product attributes, not hallucinations
- **Latency** — ingestion ms, query p50/p95/p99 (median of 3 runs)

Full definitions and formulas are in [`benchmark/README.md`](benchmark/README.md).

## Current results

### Original arena — designs 1–12 on ski_eval v1 (20 queries)

NDCG@5 overall, sorted (also visible in the heatmap above):

| Rank | Design | NDCG@5 | Notes |
|---|---|---|---|
| 1 | 11 · Fine-tuned Embeddings | 0.527 | Best on easy + cross-domain |
| 2 | 05 · SQL + FTS5 | 0.518 | Surprisingly strong; best on vague queries |
| 3 | 07 · Bayesian | 0.472 | Best on vague queries (tied with #5) |
| 4 | 01 · Knowledge Graph | 0.470 | Best on medium, weak on vague |
| 5 | 06 · Multi-Agent | 0.470 | |
| 6 | 03 · LLM-as-Judge | 0.455 | Slow but explanation-rich |
| 7 | 02 · Pure Embedding | 0.439 | |
| 8 | 04 · Hybrid (SQL+Vec) | 0.323 | Fusion underperforms either track alone |
| 9 | 08 · TF-IDF | 0.282 | |
| 10 | 10 · Ensemble / LTR | 0.258 | LTR overfits the small training signal |
| 11 | 12 · Distilled LLM | 0.186 | Slow + lossy distillation |
| 12 | 09 · Faceted Search | 0.096 | No semantic layer = brittle on natural-language queries |

The headline finding from the original arena: a fine-tuned mini-embedding model beat every off-the-shelf vector approach **and** beat a 175B-class teacher distilled into a small student. The lexical SQL+FTS5 baseline was the dark horse — cheapest design, landed second. Designs 1–12 have not been re-run against v2; results are preserved as historical baseline.

### Design 14 (local hybrid) on ski_eval v2 (100 queries)

From `artifacts/ski/eval/matrix.json` and `artifacts/ski/manifest.json`. All configs use the same Rust hot path (`arena_core`: prefilter → FTS5 + vector → RRF fusion); the variants below differ only in the embedding and rerank stages.

| Config | NDCG@5 | What's in it |
|---|---|---|
| `qwen_emb_qwen4b_listwise_vanilla` | **0.685** | Vanilla Qwen3-Embedding-0.6B + RRF + Qwen3-4B-Instruct listwise rerank |
| Shipped (LoRA emb + RRF, no rerank) | 0.614 | LoRA-tuned Qwen3-Embedding-0.6B + RRF — `manifest.json` shipped config |
| `qwen_emb_vec` | 0.579 | Vanilla Qwen3-Embedding-0.6B + RRF, no rerank |
| `lexical_only` | 0.547 | FTS5 + RRF only (no embeddings, no rerank) — the floor |

A few things this table shows:
- The lexical floor (0.547) on v2 is already roughly even with the v1 arena's #1 (0.527 fine-tuned embed). v2 is a different, harder distribution — direct cross-version comparisons aren't valid.
- LoRA fine-tuning of a 0.6B embedding lifts the no-rerank pipeline from 0.579 → 0.614 (`manifest.eval`).
- Adding a Qwen3-4B-Instruct listwise reranker on top of the vanilla embedding pushes to 0.685 — the largest single jump. Per commit `48d6dac`, "keep listwise rerank on the hot path" moved the LoRA-emb + listwise variant from 0.615 → 0.673.
- `manifest.enable_reranker: false` is intentional in the shipped config: the cross-encoder LoRA is shelved; the listwise reranker (Qwen3-1.7B LoRA) is wired up via `metadata.reranker_runtime_kind: "listwise"` but disabled in the gated config because it isn't yet a reliable lift across every query.

### Wine domain — preliminary

The wine bundle ships with a built embedding LoRA, listwise adapter, and 30-query eval graded by `openrouter/owl-alpha`. The eval gate threshold is dropped to **0.30** (vs. ski's 0.55) per `manifest.metadata.threshold_note`: the filter parser is wine-incomplete (price / points / negation hooks were the focus of the Layer 1 / Layer 2 filter-parser commits — see git log). Per commit `12521f6`, the current wine gate sits at NDCG@5 ≈ **0.331**; `manifest.eval.full_pipeline_ndcg5` is still `null`, so no canonical number has been written back yet.

`artifacts/wine/filter_phrases.json` is the Layer-2 LLM-discovered phrase map (e.g. `"patio sipper" → Pinot Gris/Rosé`, `"impress a sommelier" → points ≥ 95`). It's loaded by the runtime parser alongside the hand-curated tables.

## Running it

Python 3.9+. Some designs require a local Ollama for embeddings or LLM calls.

```bash
# Full benchmark across every design
python benchmark/runner.py --recommenders implementations/

# A subset
python benchmark/runner.py --recommenders implementations/ --filter "design_05_sql,design_11_finetuned_embed"

# Stable latency numbers
python benchmark/runner.py --recommenders implementations/ --runs 5
```

Results land in `benchmark/results/`:
- `summary.txt` / `summary.json` — comparison tables
- `per_query/{design}_{query_id}.json` — what each design returned for each query, alongside the ground truth

## Regenerating the thumbnail

```bash
python scripts/build_heatmap.py
```

Reads from `benchmark/results/` and rewrites `docs/thumbnail.png`.

## Status

Active. The shipping configuration is **design 14 (local-first hybrid)** with the domain compiler producing `artifacts/<domain>/` bundles. Two domains shipped: **ski** (gated at NDCG@5 ≥ 0.55, currently 0.614) and **wine** (gated at 0.30, currently ≈ 0.331). Designs 1–12 are preserved as the historical arena and are not on the active development path; their numbers are against the v1 ski eval and have not been re-run against v2. A Streamlit frontend (`frontend/app.py`) is wired to both domain bundles.

---

## Framework: design 13, design 14, and the domain compiler

Two new designs landed on top of the original twelve, plus a framework for producing per-domain recommender bundles.

### Design 13 — Frontier Listwise (Opus 4.7)

The arena's **upper-bound reference**. Stuffs the entire per-domain catalog (products + every review) into one prompt with cache-control and asks Opus 4.7 to return the top-k product IDs as a JSON array. Predicted NDCG@5: 0.72–0.78. Cost: ~$0.05/query with prompt caching, ~$1.40 per full benchmark run. Not a production design — it exists to anchor the gap between "frontier with full context" and "what a local pipeline can do".

See [`designs/design-13-opus-listwise.md`](designs/design-13-opus-listwise.md).

### Design 14 — Local-First Hybrid

The shipping design. Pipeline:

```
query → filter parser → hard prefilter (Rust) → parallel FTS5 (Rust)
                                              + vector encoder (Python)
        → RRF fusion (Rust) → optional listwise rerank (Python) → explanation
```

Hot path is in **Rust** via [`arena_core/`](arena_core/) (PyO3 + rusqlite/bundled SQLite). ML model invocations stay in Python. Three Rust kernels: `rrf_fuse` (the design-4 score-fusion fix), `build_prefilter_sql` (injection-safe SQL assembly), `fts5_search` (BM25 retrieval), plus `hard_negative_mine` for fast contrastive-pair generation during fine-tuning.

**Filter parser** has two layers:
- Layer 1 (commit `caf11f5`): generic catalog-derived hooks. Currency-anchored price extractors, points/pts extractors, quality-tier phrases via catalog quantiles, categorical-value mentions across all text attrs, word-level fallback for multi-word values, and clause-aware negation.
- Layer 2 (commit `12521f6`): LLM-discovered phrase mappings. An opt-in `discover-filters` step calls the teacher LLM with the catalog + sample queries and saves a JSON map of domain phrases → filter constraints to `artifacts/<domain>/filter_phrases.json`.

**Reranker** is a Qwen3-1.7B listwise LoRA (`shared/listwise_reranker.py`). The cross-encoder LoRA is shelved; the runtime picks listwise via `manifest.metadata.reranker_runtime_kind`. `manifest.enable_reranker` gates whether it runs in the shipped config (currently `false` for ski — the gain isn't uniform).

Measured numbers on ski_eval v2 (see the [Current results](#current-results) table for the full breakdown):

| Config | NDCG@5 |
|---|---|
| Vanilla Qwen3-Embedding-0.6B + RRF + Qwen3-4B-Instruct listwise rerank | 0.685 |
| Shipped: LoRA Qwen3-Embedding-0.6B + RRF (no rerank) | 0.614 |
| Vanilla Qwen3-Embedding-0.6B + RRF | 0.579 |
| Lexical only (FTS5 + RRF) | 0.547 |

See [`designs/design-14-local-first-hybrid.md`](designs/design-14-local-first-hybrid.md) for the spec and [`implementations/design_14_local_hybrid/`](implementations/design_14_local_hybrid/) for the runtime.

### The domain compiler — `arena new <domain>`

The framework that produces reusable per-domain bundles. Each bundle is a self-contained directory holding everything needed to run a recommender on one product domain: SQLite + FTS5 index, embedding model (vanilla or fine-tuned), optional reranker LoRA, filter schema, and an eval set. Same runtime, swap domains by config.

```bash
# Build the Rust core once
cd arena_core && pip install maturin && maturin build --release
pip install target/wheels/arena_core-*.whl && cd ..

# Compile a bundle from raw data
python scripts/arena_new.py new ski \
    --catalog benchmark/data/ski_products.json \
    --reviews benchmark/data/ski_reviews.json \
    --eval    benchmark/data/per_domain/ski_eval_v2.json \
    --steps ingest,embedding-pin,reranker-pin,eval-import,generate-synthetic,finetune-embedding,finetune-reranker,finetune-listwise

# Apply the ship/no-ship gate (blocks publish if NDCG < threshold)
python scripts/eval_bundle.py --bundle artifacts/ski

# Full eval matrix across configurations (writes artifacts/<domain>/eval/matrix.json)
python scripts/eval_matrix.py --bundle artifacts/ski

# Pack for distribution ("send my friend a recommender")
python scripts/arena_new.py pack artifacts/ski --out ski.tar.gz

# Unpack on the receiver's box
python scripts/arena_new.py unpack ski.tar.gz --into artifacts
```

Step-by-step scripts under [`scripts/`](scripts/):
- `arena_new.py` — the orchestrator + `inspect` / `pack` / `unpack` subcommands
- `generate_synthetic_queries.py` — teacher LLM generates query/positive/hard-negative triples; uses `arena_core.hard_negative_mine` for fast NN over the catalog
- `discover_filter_phrases.py` — Layer-2 filter parser: teacher LLM proposes domain phrase → constraint mappings, written to `artifacts/<domain>/filter_phrases.json`
- `finetune_embedding.py` — sentence-transformers `MultipleNegativesRankingLoss` with hard negatives, LoRA or full
- `finetune_reranker.py` — PEFT LoRA on `bge-reranker-v2-m3` (cross-encoder track; currently shelved)
- `finetune_listwise.py` — PEFT LoRA on Qwen3-1.7B for listwise reranking (the active reranker track)
- `eval_bundle.py` — the **ship/no-ship gate**. Refuses to ship a bundle that doesn't beat lexical-only and the per-domain `manifest.eval.arena_threshold_ndcg5` (0.55 ski, 0.30 wine), or that regresses any previously-passing query.
- `eval_matrix.py` — runs a fixed set of configs against the bundle's eval set, writes `artifacts/<domain>/eval/matrix.json` for the table above.

Bundle data model in [`shared/domain_bundle.py`](shared/domain_bundle.py); inference tier router (local / hybrid / frontier) in [`shared/inference_tiers.py`](shared/inference_tiers.py).

The framework genuinely transfers across domains. Wine (200 products from HF wine-reviews, 30-query eval) ships in `artifacts/wine/` and a small books dataset under [`benchmark/data/third_domain/`](benchmark/data/third_domain/) is the next target — `arena new book ...` produces a working recommender with no code changes.

### Frontend

A Streamlit UI lives at [`frontend/app.py`](frontend/app.py) and renders results from any built bundle. It auto-discovers domains under `artifacts/`, exposes a top-K slider and a listwise-rerank toggle, and produces domain-specific cards (wine: variety/region/points/price; ski: terrain/stiffness/waist).

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```
