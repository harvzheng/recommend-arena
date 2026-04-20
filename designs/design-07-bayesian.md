# Design 07: Bayesian / Probabilistic Scoring

A domain-agnostic recommendation system that treats product-preference matching as probabilistic inference. Each product maintains posterior distributions over structured attributes, updated from review evidence. User queries become observations in the same probabilistic space, and ranking reduces to computing P(match | evidence).

---

## 1. Architecture Overview

The system is built on a two-phase Bayesian framework:

**Phase 1 -- Belief Formation (Ingestion).** For every product, we maintain a probability distribution over each attribute (e.g., "stiffness" for skis, "cushion" for shoes). Reviews are evidence that updates these distributions via Bayes' rule. A ski with 40 reviews calling it "stiff" and 3 calling it "flexible" produces a posterior heavily concentrated on the stiff end. We do not collapse this to a point estimate -- the full distribution is the representation.

**Phase 2 -- Belief Querying (Ranking).** A user query like "stiff, on-piste carving ski" is parsed into a set of desired attribute values (observations). For each product, we compute the posterior probability that the product satisfies all stated preferences simultaneously, marginalizing over attributes the user did not mention. Products are ranked by this posterior, and uncertainty bands communicate confidence.

The key insight: the LLM handles everything linguistic (parsing reviews, parsing queries, mapping to attribute schemas), while all ranking and scoring is pure math -- transparent, debuggable, and deterministic given the same distributions.

```
Reviews --> [LLM Extraction] --> Attribute Evidence
                                      |
                                      v
                              Bayesian Update (per product)
                                      |
                                      v
                              Posterior Distributions (stored)

User Query --> [LLM Parse] --> Desired Attribute Values
                                      |
                                      v
                              Posterior Match Score (per product)
                                      |
                                      v
                              Ranked Results + Uncertainty
```

## 2. Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Bayesian inference | **pgmpy** + **scipy.stats** | pgmpy for graphical model structure when attributes are correlated; scipy for fast conjugate updates on independent attributes |
| LLM extraction | **Claude API** (or local llama.cpp) | Structured output for attribute extraction from reviews and query parsing |
| Storage | **SQLite + JSON** | Product metadata in SQLite; posterior distribution parameters serialized as JSON blobs |
| API layer | **FastAPI** | Lightweight local-first server |
| Language | **Python 3.11+** | Ecosystem alignment |

We intentionally avoid PyMC/Stan for the core loop. Full MCMC is overkill when most attributes admit conjugate priors (Beta-Binomial for binary/ordinal, Dirichlet-Multinomial for categorical, Normal-Normal for continuous). We reserve pgmpy for modeling inter-attribute dependencies when a domain requires it (e.g., ski stiffness correlates with turn radius).

## 3. Data Model

### 3.1 Domain Schema

Each domain defines an attribute schema -- a list of attributes, their types, and their priors.

```python
@dataclass
class AttributeSpec:
    name: str                          # e.g., "stiffness"
    attr_type: Literal["ordinal", "categorical", "continuous"]
    levels: list[str] | None           # for ordinal/categorical: ["soft", "medium", "stiff"]
    prior: dict                        # prior hyperparameters

# Example: ski domain
SKI_SCHEMA = [
    AttributeSpec("stiffness", "ordinal", ["soft", "medium", "stiff"],
                  prior={"alpha": [1, 1, 1]}),  # Dirichlet uniform
    AttributeSpec("terrain", "categorical", ["on-piste", "all-mountain", "backcountry", "park"],
                  prior={"alpha": [1, 1, 1, 1]}),
    AttributeSpec("waist_width_mm", "continuous", None,
                  prior={"mu": 90, "sigma": 15}),  # Normal prior
]
```

### 3.2 Product Belief State

Each product stores posterior hyperparameters per attribute -- not samples, not point estimates.

```python
@dataclass
class ProductBelief:
    product_id: str
    domain: str
    name: str
    posteriors: dict[str, dict]  # attribute_name -> posterior hyperparameters
    evidence_count: int          # total review observations incorporated

# After ingesting 40 "stiff" + 5 "medium" + 3 "soft" reviews:
# posteriors["stiffness"] = {"alpha": [4, 6, 41]}  # Dirichlet posterior
```

### 3.3 SQLite Schema

```sql
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    name TEXT NOT NULL,
    posteriors JSON NOT NULL,      -- serialized posterior hyperparameters
    evidence_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP
);

CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    product_id TEXT REFERENCES products(product_id),
    raw_text TEXT,
    extracted JSON,               -- LLM extraction result
    ingested_at TIMESTAMP
);

CREATE TABLE domain_schemas (
    domain TEXT PRIMARY KEY,
    schema JSON NOT NULL           -- list of AttributeSpec
);
```

## 4. Ingestion Pipeline

Ingestion converts raw review text into posterior updates.

**Step 1: LLM Extraction.** Each review is sent to the LLM with the domain schema and a structured output prompt. The LLM returns attribute observations.

```python
async def extract_attributes(review_text: str, schema: list[AttributeSpec]) -> dict:
    prompt = f"""Extract product attributes from this review.
    Schema: {json.dumps([s.__dict__ for s in schema])}
    Review: {review_text}
    Return JSON mapping attribute names to observed values.
    Only include attributes clearly mentioned. Use exact level names."""

    response = await llm.structured_output(prompt)
    return response  # e.g., {"stiffness": "stiff", "terrain": "on-piste"}
```

**Step 2: Bayesian Update.** Each extracted attribute observation updates the product's posterior. For conjugate families this is arithmetic, not MCMC.

```python
def update_posterior(belief: ProductBelief, observation: dict, schema: list[AttributeSpec]):
    for attr_name, observed_value in observation.items():
        spec = get_spec(schema, attr_name)
        posterior = belief.posteriors[attr_name]

        if spec.attr_type in ("ordinal", "categorical"):
            # Dirichlet-Multinomial: increment the observed category's alpha
            idx = spec.levels.index(observed_value)
            posterior["alpha"][idx] += 1

        elif spec.attr_type == "continuous":
            # Normal-Normal conjugate update
            mu_0, sigma_0 = posterior["mu"], posterior["sigma"]
            x = float(observed_value)
            sigma_obs = spec.prior.get("obs_sigma", 5.0)
            precision_0 = 1 / sigma_0**2
            precision_obs = 1 / sigma_obs**2
            posterior["mu"] = (precision_0 * mu_0 + precision_obs * x) / (precision_0 + precision_obs)
            posterior["sigma"] = (1 / (precision_0 + precision_obs)) ** 0.5

    belief.evidence_count += 1
```

**Step 3: Persist.** Write the updated `posteriors` JSON back to SQLite. The entire update for one review is sub-millisecond -- no sampling needed.

Batch ingestion processes reviews sequentially per product (order-invariant for conjugate updates) and can parallelize across products.

## 5. Query Pipeline

A user query follows a mirror path: parse, then score.

**Step 1: Parse Query.** The LLM maps natural language to attribute constraints.

```python
async def parse_query(query: str, schema: list[AttributeSpec]) -> list[AttributeConstraint]:
    prompt = f"""Parse this product query into attribute constraints.
    Schema: {json.dumps([s.__dict__ for s in schema])}
    Query: "{query}"
    Return list of {{"attribute": str, "value": str, "strength": float}}.
    strength is 0-1: how important this constraint seems."""

    return await llm.structured_output(prompt)
    # e.g., {"mapped": [{"attribute": "stiffness", "value": "stiff", "strength": 0.9},
    #                    {"attribute": "terrain", "value": "on-piste", "strength": 0.8}],
    #         "unmapped": ["playful", "fun"]}
```

**Step 1b: Unmapped-Term Fallback.** When the LLM query parser cannot map a term to a schema attribute (e.g., "playful" or "fun" in a ski domain), it flags these terms as `unmapped`. Rather than silently dropping them -- which would cause zero-result failures for freeform queries -- we fall back to a simple text search against stored review text.

```python
def text_search_fallback(unmapped_terms: list[str], product_ids: list[str],
                         db: sqlite3.Connection) -> dict[str, float]:
    """Score products by how often unmapped terms appear in their reviews."""
    scores = {}
    for pid in product_ids:
        reviews = db.execute(
            "SELECT raw_text FROM reviews WHERE product_id = ?", (pid,)
        ).fetchall()
        corpus = " ".join(r[0].lower() for r in reviews)
        term_hits = sum(1 for t in unmapped_terms if t.lower() in corpus)
        # Normalize: fraction of unmapped terms found in this product's reviews
        scores[pid] = term_hits / len(unmapped_terms) if unmapped_terms else 0.0
    return scores
```

The fallback score is combined with the Bayesian match score as a weighted sum: `final = (1 - w_fallback) * bayesian_score + w_fallback * fallback_score`, where `w_fallback` is proportional to the fraction of query terms that were unmapped. If all terms map to schema attributes, the fallback has zero weight; if none map, the fallback drives the ranking entirely. This prevents the hard failure mode where a freeform query like "something playful and fun" returns nothing because those terms are not in the schema.

**Step 2: Compute Match Score.** For each product, compute the posterior probability of matching every stated constraint, weighted by strength.

```python
def score_product(belief: ProductBelief, constraints: list[dict],
                  schema: list[AttributeSpec]) -> tuple[float, float]:
    log_score = 0.0
    total_variance = 0.0

    for c in constraints:
        spec = get_spec(schema, c["attribute"])
        posterior = belief.posteriors[c["attribute"]]
        strength = c["strength"]

        if spec.attr_type in ("ordinal", "categorical"):
            # Probability of the desired level under Dirichlet posterior
            alpha = posterior["alpha"]
            idx = spec.levels.index(c["value"])
            p = alpha[idx] / sum(alpha)
            # Variance of this estimate (Beta variance for this component)
            alpha_0 = sum(alpha)
            var = (alpha[idx] * (alpha_0 - alpha[idx])) / (alpha_0**2 * (alpha_0 + 1))
            log_score += strength * math.log(p + 1e-10)
            total_variance += strength**2 * var

        elif spec.attr_type == "continuous":
            # CDF-based interval scoring: compute the probability that the
            # product's true attribute value lies within [target - epsilon,
            # target + epsilon] under the Normal posterior. This yields a
            # proper probability (unlike raw Gaussian proximity) that is
            # comparable to the categorical/ordinal probabilities.
            from scipy.stats import norm
            mu, sigma = posterior["mu"], posterior["sigma"]
            target = float(c["value"])
            epsilon = spec.prior.get("epsilon", spec.prior.get("obs_sigma", 5.0))
            p = norm.cdf(target + epsilon, mu, sigma) - norm.cdf(target - epsilon, mu, sigma)
            log_score += strength * math.log(p + 1e-10)
            total_variance += strength**2 * sigma**2

    score = math.exp(log_score)
    uncertainty = math.sqrt(total_variance)
    return score, uncertainty
```

The score is a product of per-attribute match probabilities (weighted by importance). Uncertainty propagates from the width of each posterior -- a product with 3 reviews will have wide posteriors and high uncertainty; one with 300 reviews will be precise.

**Score Normalization.** Raw match scores are products of probabilities and can be extremely small (e.g., 0.0003 for a 5-attribute query). For external consumption and benchmark comparability, we normalize scores to the 0-1 range using min-max normalization across all candidate products for a given query:

```python
def normalize_scores(scored: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Min-max normalize raw scores to [0, 1] across all candidates."""
    raw_scores = [s[1] for s in scored]
    s_min, s_max = min(raw_scores), max(raw_scores)
    if s_max - s_min < 1e-12:
        # All scores identical -- normalize to 1.0
        return [(pid, 1.0, unc) for pid, _, unc in scored]
    return [(pid, (raw - s_min) / (s_max - s_min), unc)
            for pid, raw, unc in scored]
```

This ensures the top-ranked product always scores 1.0 and the worst-ranked scores 0.0, regardless of how many attributes the query touches. The normalization is applied after scoring but before ranking and is required for the benchmark's `RecommendationResult.score` field.

## 6. Ranking Strategy

Products are ranked by posterior match score. The system returns three pieces of information per result:

1. **Score** -- P(match | reviews, query). Higher is better.
2. **Uncertainty** -- How confident we are. Derived from posterior variance.
3. **Evidence count** -- Raw number of reviews incorporated.

Ranking modes:

- **Expected value** (default): rank by score directly. Best for users who want the most likely match.
- **Optimistic (UCB)**: rank by `score + k * uncertainty`. Surfaces promising products with few reviews. Useful for discovery.
- **Conservative (LCB)**: rank by `score - k * uncertainty`. Only surfaces products we are confident about.

```python
def rank_products(scored: list[tuple[str, float, float]],
                  mode: str = "expected", k: float = 1.0) -> list:
    if mode == "expected":
        return sorted(scored, key=lambda x: x[1], reverse=True)
    elif mode == "optimistic":
        return sorted(scored, key=lambda x: x[1] + k * x[2], reverse=True)
    elif mode == "conservative":
        return sorted(scored, key=lambda x: x[1] - k * x[2], reverse=True)
```

The `ranking_mode` and `k` parameters are exposed through the common `query()` interface as optional keyword arguments (see Section 10). The benchmark harness can exercise all three modes.

**Concrete example -- UCB vs LCB in practice:**

Consider querying "stiff on-piste ski" with these candidates:

| Product | Score | Uncertainty | Reviews | EV Rank | UCB (k=1) | LCB (k=1) |
|---------|-------|-------------|---------|---------|-----------|-----------|
| Nordica Enforcer 100 | 0.82 | 0.03 | 180 | 2 | 0.85 (rank 2) | 0.79 (rank 1) |
| Fischer RC4 | 0.85 | 0.02 | 250 | 1 | 0.87 (rank 1) | 0.83 (rank 1) |
| Kastle MX88 | 0.70 | 0.18 | 8 | 3 | 0.88 (rank 1) | 0.52 (rank 3) |

Under **UCB** ranking, the Kastle MX88 -- a niche ski with only 8 reviews -- surfaces to the top. Its high uncertainty inflates its score, giving it a chance to be discovered. This is ideal for adventurous users or for identifying under-reviewed products that might be excellent.

Under **LCB** ranking, the Fischer RC4 dominates because its narrow uncertainty (250 reviews) means we are very confident in its score. The Kastle drops to last because we cannot be sure about it. This is ideal for conservative users who want safe recommendations.

Results are presented with confidence indicators: a product scoring 0.85 with uncertainty 0.02 (200 reviews) is a confident recommendation. A product scoring 0.88 with uncertainty 0.15 (6 reviews) is flagged as promising but uncertain.

**Explanation Generation.** The probabilistic framework produces naturally interpretable explanations. Each score traces directly to posterior probabilities derived from review counts, making explanations concrete and evidence-based.

```python
def build_explanation(belief: ProductBelief, constraints: list[dict],
                      schema: list[AttributeSpec]) -> tuple[str, dict[str, float]]:
    """Build a human-readable explanation and per-attribute score breakdown."""
    from scipy.stats import norm
    parts = []
    matched_attributes = {}

    for c in constraints:
        spec = get_spec(schema, c["attribute"])
        posterior = belief.posteriors[c["attribute"]]

        if spec.attr_type in ("ordinal", "categorical"):
            alpha = posterior["alpha"]
            idx = spec.levels.index(c["value"])
            p = alpha[idx] / sum(alpha)
            obs_count = int(alpha[idx] - 1)  # subtract prior
            total_obs = int(sum(alpha) - len(alpha))  # subtract all priors
            parts.append(f'P({c["value"]}) = {p:.2f} (from {obs_count}/{total_obs} reviews)')
            matched_attributes[c["attribute"]] = round(p, 4)

        elif spec.attr_type == "continuous":
            mu, sigma = posterior["mu"], posterior["sigma"]
            target = float(c["value"])
            epsilon = spec.prior.get("epsilon", spec.prior.get("obs_sigma", 5.0))
            p = norm.cdf(target + epsilon, mu, sigma) - norm.cdf(target - epsilon, mu, sigma)
            parts.append(f'P({c["attribute"]} in [{target-epsilon:.0f}, {target+epsilon:.0f}]) '
                         f'= {p:.2f} (posterior mu={mu:.1f}, sigma={sigma:.1f})')
            matched_attributes[c["attribute"]] = round(p, 4)

    # Confidence label based on evidence count
    n = belief.evidence_count
    if n >= 50:
        confidence = "high"
    elif n >= 15:
        confidence = "medium"
    else:
        confidence = "low"

    explanation = (
        f"Recommended because {', '.join(parts)}. "
        f"Confidence: {confidence} ({n} total observations)."
    )
    return explanation, matched_attributes
```

Example output: *"Recommended because P(stiff) = 0.95 (from 40/43 reviews), P(on-piste) = 0.88 (from 35/40 reviews). Confidence: high (200 total observations)."*

## 7. Domain Adaptation

Adding a new domain requires exactly one artifact: an attribute schema with priors.

```python
RUNNING_SHOE_SCHEMA = [
    AttributeSpec("cushion", "ordinal", ["minimal", "moderate", "maximal"],
                  prior={"alpha": [1, 1, 1]}),
    AttributeSpec("stability", "categorical", ["neutral", "guidance", "motion-control"],
                  prior={"alpha": [1, 1, 1]}),
    AttributeSpec("has_plate", "categorical", ["yes", "no"],
                  prior={"alpha": [1, 1]}),
    AttributeSpec("weight_oz", "continuous", None,
                  prior={"mu": 10, "sigma": 3, "obs_sigma": 1.0}),
    AttributeSpec("drop_mm", "continuous", None,
                  prior={"mu": 8, "sigma": 4, "obs_sigma": 1.5}),
]
```

**Informative priors** can encode domain knowledge: if 80% of skis are medium-stiff, set `alpha = [2, 8, 8]` instead of uniform. This helps cold-start products converge faster. Uniform priors (all 1s) are safe defaults when domain knowledge is absent.

**Schema discovery** can itself be LLM-assisted: feed 20 sample reviews to the LLM and ask it to propose an attribute schema. A human reviews and locks it before ingestion begins.

**Correlation structure.** When attributes are correlated (cushion and weight in shoes), pgmpy models this as a Bayesian network with edges between correlated attributes. The update step then uses belief propagation instead of independent conjugate updates. This is optional -- independent attributes work well for most domains and are far simpler.

## 8. Pros and Cons

### Strengths

- **Principled uncertainty.** Every score carries a confidence measure. The system knows what it does not know, unlike embedding-similarity approaches that produce confident-looking scores regardless of evidence quality.
- **Interpretable.** "This ski scored 0.91 because P(stiff) = 0.95 from 40 reviews and P(on-piste) = 0.88 from 35 reviews." Every number traces to evidence. No black-box embeddings.
- **Incremental updates.** Conjugate priors mean each new review is a constant-time arithmetic update. No retraining, no reindexing, no batch jobs.
- **Cold-start handling.** Products with few reviews naturally get wide posteriors and high uncertainty. The ranking modes (UCB/LCB) let users choose how to handle this.
- **Domain-agnostic.** Attribute schemas are the only domain-specific artifact. The math is identical across skis, shoes, and cookies.
- **Deterministic ranking.** Given the same posteriors and query, scores are identical every time. No stochastic embedding drift.

### Weaknesses

- **Schema rigidity.** Attributes must be pre-defined. A review mentioning a novel attribute (e.g., "great edge hold on ice") is lost unless the schema includes it. Mitigation: periodic schema review, or a catch-all "tags" attribute.
- **Independence assumption.** Treating attributes as independent loses correlations (stiff skis are usually narrower). pgmpy can model this but adds significant complexity.
- **LLM extraction bottleneck.** The system is only as good as the LLM's ability to map "this thing is a rocket ship on hardpack" to `{"stiffness": "stiff", "terrain": "on-piste"}`. Extraction errors compound in the posterior.
- **No free-text similarity.** A query like "something similar to the Nordica Enforcer" has no natural representation. This is a structured-attribute system, not a vibes-based one. Mitigated by the unmapped-term fallback (see Section 5, Step 1b) which falls back to text search against stored review text when query terms cannot be mapped to schema attributes.
- **Ordinal coarseness.** Mapping continuous concepts (stiffness is really a spectrum) to 3-5 ordinal levels loses resolution. More levels means more data needed to fill each bucket.

## 9. POC Scope

A minimal proof-of-concept for the ski domain.

**Scope:** 5 ski products, 10-20 synthetic reviews each, 3 attributes (stiffness, terrain, waist width). CLI interface for queries.

```python
# poc.py -- end-to-end sketch
import json, math, sqlite3
from dataclasses import dataclass, field, asdict

SCHEMA = [
    {"name": "stiffness", "type": "ordinal", "levels": ["soft", "medium", "stiff"],
     "prior": {"alpha": [1, 1, 1]}},
    {"name": "terrain", "type": "categorical",
     "levels": ["on-piste", "all-mountain", "backcountry", "park"],
     "prior": {"alpha": [1, 1, 1, 1]}},
    {"name": "waist_mm", "type": "continuous", "levels": None,
     "prior": {"mu": 90, "sigma": 15, "obs_sigma": 5}},
]

def init_posteriors(schema):
    return {s["name"]: dict(s["prior"]) for s in schema}

def update(posteriors, observation, schema):
    schema_map = {s["name"]: s for s in schema}
    for attr, value in observation.items():
        spec = schema_map[attr]
        post = posteriors[attr]
        if spec["type"] in ("ordinal", "categorical"):
            idx = spec["levels"].index(value)
            post["alpha"][idx] += 1
        elif spec["type"] == "continuous":
            mu0, s0 = post["mu"], post["sigma"]
            s_obs = spec["prior"]["obs_sigma"]
            prec0, prec_obs = 1/s0**2, 1/s_obs**2
            post["mu"] = (prec0*mu0 + prec_obs*float(value)) / (prec0+prec_obs)
            post["sigma"] = (1/(prec0+prec_obs))**0.5

def score(posteriors, constraints, schema):
    schema_map = {s["name"]: s for s in schema}
    log_s = 0.0
    for c in constraints:
        spec = schema_map[c["attribute"]]
        post = posteriors[c["attribute"]]
        w = c.get("strength", 1.0)
        if spec["type"] in ("ordinal", "categorical"):
            idx = spec["levels"].index(c["value"])
            p = post["alpha"][idx] / sum(post["alpha"])
            log_s += w * math.log(p + 1e-10)
        elif spec["type"] == "continuous":
            # CDF-based interval scoring (proper probability)
            from scipy.stats import norm as sp_norm
            target = float(c["value"])
            epsilon = spec["prior"].get("epsilon", spec["prior"].get("obs_sigma", 5.0))
            p = sp_norm.cdf(target + epsilon, post["mu"], post["sigma"]) \
                - sp_norm.cdf(target - epsilon, post["mu"], post["sigma"])
            log_s += w * math.log(p + 1e-10)
    return math.exp(log_s)

# --- Demo ---
products = {
    "nordica-enforcer": {"stiffness": "stiff", "terrain": "on-piste", "waist_mm": "88"},
    "blizzard-rustler": {"stiffness": "medium", "terrain": "all-mountain", "waist_mm": "96"},
}

# Simulate ingestion
beliefs = {}
for pid, typical in products.items():
    beliefs[pid] = init_posteriors(SCHEMA)
    for _ in range(30):  # 30 reviews agreeing
        update(beliefs[pid], typical, SCHEMA)

# Query: stiff on-piste ski
query = [
    {"attribute": "stiffness", "value": "stiff", "strength": 0.9},
    {"attribute": "terrain", "value": "on-piste", "strength": 0.8},
]

for pid in beliefs:
    s = score(beliefs[pid], query, SCHEMA)
    print(f"{pid}: {s:.4f}")
# Expected: nordica-enforcer >> blizzard-rustler
```

**POC milestones:**

1. Hard-coded schema + synthetic reviews, pure math scoring (day 1)
2. Wire LLM extraction for real review text (day 2)
3. LLM-based query parsing (day 2)
4. SQLite persistence + FastAPI endpoint (day 3)
5. Add uncertainty display and ranking modes (day 3)

**POC ranking mode demonstration:**

```python
# After scoring all products, demonstrate UCB vs LCB
scored = []
for pid in beliefs:
    s, unc = score_with_uncertainty(beliefs[pid], query, SCHEMA)
    scored.append((pid, s, unc))

scored = normalize_scores(scored)

print("=== Expected Value Ranking ===")
for pid, s, u in rank_products(scored, mode="expected"):
    print(f"  {pid}: score={s:.3f}, uncertainty={u:.3f}")

print("\n=== UCB Ranking (discovery mode) ===")
for pid, s, u in rank_products(scored, mode="optimistic", k=1.5):
    print(f"  {pid}: score={s + 1.5*u:.3f} (base={s:.3f}, unc={u:.3f})")

print("\n=== LCB Ranking (conservative mode) ===")
for pid, s, u in rank_products(scored, mode="conservative", k=1.5):
    print(f"  {pid}: score={s - 1.5*u:.3f} (base={s:.3f}, unc={u:.3f})")
```

**What the POC validates:** That conjugate Bayesian updates from LLM-extracted review data produce intuitive, explainable rankings that correctly separate products along user-specified preference axes -- with honest uncertainty quantification that point-estimate systems cannot provide.

## 10. Common Interface

All recommender designs implement a shared protocol so the benchmark harness can evaluate them uniformly. The Bayesian recommender conforms to this interface.

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class RecommendationResult:
    product_id: str
    product_name: str
    score: float  # 0-1 normalized (min-max across candidates for this query)
    explanation: str
    matched_attributes: dict[str, float]  # attribute_name -> per-attribute probability


class Recommender(Protocol):
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None: ...
    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]: ...
```

**Implementation mapping for this design:**

- `ingest()` loads the domain schema (or auto-discovers one), runs LLM extraction on each review, and applies Bayesian updates to build posterior beliefs per product.
- `query()` parses the query via the LLM, computes per-product match scores (including the unmapped-term fallback if needed), normalizes scores to 0-1 via min-max, builds explanations via `build_explanation()`, and returns the top-k results.
- `RecommendationResult.score` is always the min-max normalized score (see Section 5). The top result scores 1.0; the worst candidate scores 0.0.
- `RecommendationResult.explanation` is generated by `build_explanation()` (see Section 6), providing evidence-traced reasoning like *"Recommended because P(stiff) = 0.95 (from 40/43 reviews)..."*.
- `RecommendationResult.matched_attributes` contains per-attribute probabilities: for categorical/ordinal attributes this is the Dirichlet posterior probability of the desired level; for continuous attributes this is the CDF-based interval probability P(value in [target - epsilon, target + epsilon]).

**Optional parameters.** The `query()` method accepts additional keyword arguments for this design:

```python
def query(self, query_text: str, domain: str, top_k: int = 10, *,
          ranking_mode: str = "expected",  # "expected" | "optimistic" | "conservative"
          k: float = 1.0,                  # UCB/LCB multiplier
          ) -> list[RecommendationResult]: ...
```

These allow the benchmark to exercise the UCB/LCB ranking modes that are a unique feature of this design. Other recommender implementations simply ignore these parameters.
