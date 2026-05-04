# Design 13 — Frontier Listwise (Opus 4.7)

The arena's upper-bound reference. Stuff the entire per-domain catalog
(products + every review) into one cached system prompt and ask Opus 4.7 to
return the top-k product IDs as a JSON array.

See [`designs/design-13-opus-listwise.md`](../../designs/design-13-opus-listwise.md)
for the full spec, rationale, and cost model.

## Why this design exists

To anchor the gap between "frontier model with full context" and "what a
local pipeline can do". Every other design is measured against the headroom
this exposes. We don't ship this — it costs ~$0.05/query and won't scale
past ~10k products.

## Quick start

```bash
pip install -r implementations/design_13_opus_listwise/requirements.txt
export ANTHROPIC_API_KEY=...
python benchmark/runner.py --recommenders implementations/ \
    --filter "design_13_opus_listwise"
```

## What it does

1. **Ingest**: format the catalog (products + reviews, sorted deterministically)
   into one text block. No API calls.
2. **Query**: send the catalog as a `cache_control: ephemeral` system block,
   plus the user query. Parse the returned JSON array of IDs.
3. **Validate**: filter hallucinated IDs to the known catalog set and pad to
   `top_k` from a deterministic fallback so the runner always gets `top_k`.

## Cost

For the ski domain (~28k input tokens):

| Mode | Per query |
|---|---|
| First query (cache miss) | ~$0.43 |
| Subsequent (cache hit) | ~$0.05 |
| Full 20-query benchmark | **~$1.40** |

A local SQLite cache (`benchmark/results/design_13/cache.sqlite`) stores
`(query, catalog) → ranked IDs` so re-running the benchmark is free. Disable
with `ARENA_NO_CACHE=1`.

## Notes for committers

- The model occasionally emits IDs outside the catalog. `_call_api` filters
  to known IDs and pads with the next-highest non-duplicate from a fallback
  ranking. We log the rate of hallucinated IDs as a diagnostic.
- The catalog formatter is **deterministic** (sorted product order, sorted
  attribute keys, sorted reviews). Changing this kills the prompt cache.
- `thinking`, `temperature`, `top_p`, `top_k` are intentionally omitted —
  they're either removed on Opus 4.7 (would 400) or unhelpful for a
  deterministic listwise ranking task.
