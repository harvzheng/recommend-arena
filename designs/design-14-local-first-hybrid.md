# Design 14: Local-First Hybrid

**One sentence:** Hard prefilter → parallel FTS5 + vector retrieval over the
candidate set → rank fusion (RRF) → cross-encoder rerank → return.

The candidate winner. Designed to close the gap to design 13's frontier
ceiling at $0 marginal cost per query, runnable on consumer hardware.

The hot path is in **Rust** (`arena_core/`); ML model invocation stays in
Python. A Python wrapper (`implementations/design_14_local_hybrid/`) wires
the two together and exposes the standard `Recommender` factory.

---

## Why we expect this to beat the current leaderboard

- Beats #11 alone (0.527) because the cross-encoder fixes its bi-encoder
  ranking errors AND the lexical track catches FTS5-friendly queries the
  embedding misses.
- Beats #5 alone (0.518) because the embedding handles paraphrase and
  metaphor ("alive underfoot") that BM25 misses.
- Avoids the trap design #4 hit because RRF only sees ranks, not scores
  from incommensurate distributions.
- Avoids the trap design #12 hit because we never compress an LLM — we
  only use LLMs (optionally) for filter parsing and explanation.

Predicted NDCG@5: **0.60–0.65 with off-the-shelf components, 0.68–0.73
with a per-domain reranker fine-tune.** Action #2 in the build-out plan
ships the off-the-shelf version first.

---

## Architecture

```
query
  ↓
[1. filter parser]      Phase A: deterministic regex/keyword extractor
                        Phase B: Qwen3-4B-Instruct with constrained JSON
                                 (xgrammar / outlines)
  ↓
[2. hard prefilter]     RUST. SQL WHERE on extracted constraints.
                        Cuts catalog to a candidate set.
  ↓
[3. parallel retrieval over candidates]
  ├ lexical (RUST):     SQLite FTS5 with BM25         → top-100
  └ semantic (Python):  sentence-transformers encode  → top-100
  ↓
[4. rank fusion]        RUST. Reciprocal Rank Fusion (RRF, k=60).
                        Operates on RANKS, not scores — the design 4 fix.
                                                         → top-50
  ↓
[5. cross-encoder rerank]
                        bge-reranker-v2-m3 (568M, MIT). Optional LoRA.
                        Lazy-loaded; falls back to RRF order if not
                        installed. Python.            → top-10
  ↓
[6. explanation]        Phase A: deterministic ("matched: X, Y, Z").
                        Phase B: Qwen3-4B local, Together API, or Opus 4.7
                                 escalation when reranker top-1 vs top-2
                                 margin is below threshold.
```

## What's in Rust and why

The Rust crate is `arena_core/`, exposed to Python via PyO3.

| Component | In Rust because… |
|---|---|
| **RRF fusion** | Pure CPU math; on every query's hot path. The design-4 fix lives here. |
| **Hard prefilter** | SQL WHERE assembly + parameter collection; called every query. |
| **FTS5 candidate retrieval** | rusqlite call with BM25 column weights; fastest path to top-100. |

What stays in Python: anything that loads a PyTorch / MLX model
(embedding encoder, cross-encoder, optional filter LLM). The marginal
latency win from porting these to Rust is ≪ the maintenance cost.

## Inference tiers

| Tier | Filter parser | Explanation | Latency | $ per query |
|---|---|---|---|---|
| 1 — fully local | Qwen3-4B (or regex) | Qwen3-4B | ~700ms–1s | $0 |
| 2 — hybrid | Together Llama-3.1-8B | Together Llama-3.1-8B | ~1–2s | $0.001–0.005 |
| 3 — frontier escalation | Tier 2 default; Opus 4.7 when reranker margin < threshold | (same) | ~5% of queries pay frontier latency | $0.05–0.10 on escalated |

Implemented as a single `Recommender` with a `tier: Literal["local", "hybrid", "frontier"]` config knob.

## Build phases

The original task description lists six "first actions". Design 14 is built in three slices:

| Slice | What ships | Predicted NDCG@5 |
|---|---|---|
| 14.0 — off-the-shelf | FTS5 + vanilla Qwen3-Embedding-0.6B + RRF + vanilla bge-reranker | 0.60–0.65 |
| 14.1 — fine-tuned embedding | Fine-tuned Qwen3-Embedding-0.6B from `arena new` compiler | 0.66–0.70 |
| 14.2 — fine-tuned reranker | Per-domain bge-reranker-v2-m3 LoRA | 0.68–0.73 |

This PR ships **14.0** — the architecture spike. The compiler (`arena
new`) and the fine-tuned variants come in a follow-up.

## Eval gate

Before committing any new bundle or any code change to design 14:

```bash
python scripts/eval_bundle.py --domain skis
```

Must show:
- Fine-tuned embedding > vanilla embedding (else: synthetic data is bad)
- Full pipeline > current arena #1 (0.527) AND > both #11 alone and #5 alone (else: assembly is wrong)
- No regression on previously-good queries (track per-query NDCG, not just aggregate)

Refuse to ship the bundle if this gate fails. This is the single most
important safeguard — it would have caught design #10's LTR overfit and
design #12's bad distillation before they entered the arena.

## Deliberately out of scope for 14.0

- Fine-tuning (covered by 14.1, 14.2 and the `arena new` compiler).
- The full domain compiler CLI (`scripts/arena_new.py`).
- Filter-parsing LLM call. 14.0 uses a deterministic regex/keyword
  extractor over the existing per-domain attribute schema. The LLM
  parser is a clear v2 lift but isn't needed to demonstrate the
  architecture wins.
- The escalation router. 14.0 ships tier 1 only; tier 2/3 are config
  flags wired in the follow-up.
