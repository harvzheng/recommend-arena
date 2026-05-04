# Design 13: Frontier Listwise (Opus 4.7)

**One sentence:** Stuff the entire catalog (products + reviews) into a single
prompt with cache-control, ask Opus 4.7 to return the top-k product IDs as a
JSON array, parse, return.

This design is the **upper bound reference** for the arena. It exists not to
ship to production but to anchor the gap between "what's possible with frontier
models and full context" and "what's possible locally". Every other design is
measured against the headroom this exposes.

---

## Why we expect this to win

- The model sees every product and every review in a single context window.
  No retrieval lossiness, no embedding bottleneck, no rerank artifacts.
- Listwise ranking — the model can compare candidates against each other
  directly rather than scoring them in isolation.
- Opus 4.7 has the world knowledge to interpret metaphor ("alive
  underfoot"), trade-offs ("stiff but not punishing"), and negation ("not
  playful") in ways no embedding model can match.

Predicted NDCG@5: **0.72–0.78**, capped near ~0.85 by hand-graded ground-truth
noise. Cost: roughly **$0.20 per query without caching, $0.005 per query with
prompt caching** of the catalog.

## What this design does NOT do

- No fine-tuning, no domain-specific training of any kind.
- No retrieval — the catalog goes into the prompt verbatim.
- No explanations returned — we ask only for ordered IDs to keep latency low.
  An explanation pass can be added by re-prompting on just the top-k.
- Does not scale beyond ~10k products. The whole catalog must fit in context
  (200k tokens for Opus 4.7). At ~1 review per 200 tokens and 5 reviews per
  product, the practical ceiling is ~25k products.

---

## Architecture

```
ingest(products, reviews, domain):
  format products + reviews as a single text block, store in self.catalogs[domain]
  (no API calls — the catalog is sent at query time inside a cache-control block)

query(text, domain, top_k):
  POST messages.create:
    system = [{ text: catalogs[domain], cache_control: "ephemeral" }]
    user   = "Query: <text>\nReturn top-{k} product IDs as a JSON array."
  parse the JSON array → RecommendationResult list with descending pseudo-scores
```

The catalog text is wrapped in a `cache_control: ephemeral` block so the second
and subsequent queries against the same domain hit the prompt cache (5-min TTL)
at ~10% of input cost.

## Cost model

For the ski domain (25 products, 132 reviews, ~28k input tokens):

| Mode | Input cost | Output cost | Per query |
|---|---|---|---|
| First query (cache miss) | 28k × $15/M = $0.42 | 100 × $75/M ≈ $0.01 | **~$0.43** |
| Subsequent (cache hit) | 28k × $1.50/M = $0.042 | 100 × $75/M ≈ $0.01 | **~$0.05** |

For the 20-query benchmark: ~$0.43 + 19 × $0.05 = **~$1.40 total** per full run.
The README's "$5" figure assumed multiple runs and per-domain warmup.

## Implementation notes

- `claude-opus-4-7` is the model ID. Use the latest pinned version when
  results are committed.
- `max_tokens=1024` is plenty for top-10 IDs; larger only adds latency.
- Prompt-caching needs `cache_control: {"type": "ephemeral"}` on the *system*
  block that holds the catalog. Per-message caching has different semantics.
- The model occasionally emits product IDs outside the catalog. Filter to the
  known ID set and pad with the next-highest non-duplicate from a fallback
  ranking. We log the rate of hallucinated IDs as a diagnostic.

## What to do with the result

Commit the per-query output to `benchmark/results/design_13/per_query/` and
treat the aggregate NDCG@5 as the "frontier ceiling" line on every leaderboard
chart going forward. When a future local design closes the gap to within ~10%
of this ceiling, declare victory and stop optimizing.
