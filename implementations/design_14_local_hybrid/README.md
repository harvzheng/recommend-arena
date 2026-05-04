# Design 14 — Local-First Hybrid

The candidate winner. Architecture spike (slice 14.0): off-the-shelf
components only, no fine-tuning yet.

See [`designs/design-14-local-first-hybrid.md`](../../designs/design-14-local-first-hybrid.md)
for the full spec, predicted NDCG, and roadmap to slices 14.1 / 14.2.

## Pipeline

```
query
  ↓
[1. filter parser]      Phase A: deterministic regex/keyword extractor
  ↓
[2. hard prefilter]     RUST. SQL WHERE assembled by arena_core.
  ↓
[3. parallel retrieval]
  ├ lexical (RUST):     arena_core.fts5_search        → top-100
  └ semantic (Python):  Qwen3-Embedding-0.6B          → top-100
  ↓
[4. RRF]                RUST. arena_core.rrf_fuse, k=60.   → top-50
  ↓
[5. rerank]             bge-reranker-v2-m3 (optional)      → top-10
  ↓
[6. explanation]        deterministic ("matched: X, Y; BM25=...")
```

## Why Rust where it is

| Stage | Where | Why |
|---|---|---|
| Filter parser | Python | Rules table; ~100µs end-to-end. Rust would be a wash. |
| **Prefilter SQL assembly** | **Rust** | Hot path. Injection-safe by construction. |
| **FTS5 candidate retrieval** | **Rust** | Hot path. rusqlite + bundled SQLite. |
| Vector encode | Python | sentence-transformers / PyTorch. |
| **RRF fusion** | **Rust** | Pure CPU math. The design-4 fix. |
| Cross-encoder rerank | Python | bge-reranker is PyTorch. |
| Explanation | Python | Cheap formatting. |

Anything that loads a PyTorch model stays in Python. Everything else
that runs on every query is in Rust.

## Quick start

```bash
# Build the Rust core
cd arena_core
pip install maturin
maturin build --release
pip install target/wheels/arena_core-*.whl

# ML deps
cd ..
pip install -r implementations/design_14_local_hybrid/requirements.txt

# Run the benchmark filter
python benchmark/runner.py --recommenders implementations/ \
    --filter "design_14_local_hybrid"
```

## Graceful degradation

The recommender works in any of these states:

| `arena_core` (Rust) | `sentence-transformers` | `bge-reranker` | What runs |
|---|---|---|---|
| ✅ | ✅ | ✅ | Full pipeline, predicted NDCG@5 0.60–0.65 |
| ✅ | ✅ | ❌ | RRF order returned; no cross-encoder |
| ✅ | ❌ | — | Lexical-only (similar to #5 / 0.518) |
| ❌ | ✅ | ✅ | Falls back to Python RRF; full pipeline still runs |
| ❌ | ❌ | — | Pure-Python RRF over the candidate set, no retrieval |

The benchmark runner can discover the recommender in any of these
states without crashing on import. Whether the eval gate passes is a
separate question — see `scripts/eval_bundle.py` (slice 14.1).

## Tunables

In `recommender.py`:

```python
TOP_K_LEXICAL = 100   # FTS5 top-N before fusion
TOP_K_VECTOR  = 100   # vector top-N before fusion
TOP_K_FUSED   = 50    # RRF output → fed to reranker
TOP_K_RERANKED = 10   # cross-encoder output → returned
RRF_K = 60            # RRF dampening constant (Cormack et al. 2009)
```

## Slice roadmap

| Slice | Adds | Predicted NDCG@5 |
|---|---|---|
| **14.0 (this PR)** | off-the-shelf | 0.60–0.65 |
| 14.1 | fine-tuned Qwen3-Embedding-0.6B from `arena new` | 0.66–0.70 |
| 14.2 | fine-tuned bge-reranker LoRA per domain | 0.68–0.73 |
