# Design Enhancement Spec: Cheap Signal Boosting

**Goal:** Push NDCG@5 toward 0.8 across top 4 designs using only cheap/free signals (no additional LLM calls).

**Constraint:** Enhancements must use embeddings, dictionaries, deterministic logic, or improved prompts. No new LLM calls beyond what each design already makes.

**Target Designs:** design_00_sota (0.535), design_05_sql (0.511), design_07_bayesian (0.499), design_01_graph (0.470)

## Shared Enhancements (all designs)

1. **Domain synonym dictionary** — deterministic query expansion before LLM parse
2. **Few-shot examples in prompts** — better parse quality at same cost
3. **Attribute schema seeding** — known attributes per domain instead of inference

## Per-Design Enhancements

### design_00_sota
- TF-IDF keyword matching from reviews as third scoring signal
- Near-miss decay scoring for attributes
- Seed attribute schemas per domain

### design_05_sql
- Embedding cosine similarity as third signal (alongside BM25 + attributes)
- Expand synonym dictionary coverage
- Pre-expand vague terms via synonyms before LLM parse

### design_07_bayesian
- Batch reviews per product (reduce LLM calls)
- Embedding signal for vague query fallback
- Stronger domain priors from known attribute ranges

### design_01_graph
- Embedding-based semantic edges/scoring
- Near-miss tolerance in graph scoring
- Better attribute coverage via schema seeding
