# Design #9: Faceted Search + Elasticsearch-Style Recommendation

**Design philosophy:** Treat recommendation as faceted search. Extract structured facets from unstructured reviews, index them in a purpose-built search engine (Typesense), and let the search engine's filtering, text matching, and custom ranking do the heavy lifting. The LLM's role is strictly limited to two tasks: extracting facets from reviews at ingestion time, and translating natural language queries into facet filters at query time. Everything else is deterministic search infrastructure.

---

## 1. Architecture Overview

```
                        INGESTION
                           |
             Raw Reviews (any domain)
                           |
                   LLM Facet Extraction
                           |
                 Aggregate per product
                           |
                  Typesense Indexing
                    (faceted docs)

                         QUERY
                           |
                  "stiff on-piste ski, 180cm+"
                           |
                   LLM Query Parser
                      /         \
            Facet Filters     Text Query
                      \         /
                  Typesense Search
                (filter + rank + sort)
                           |
                    Ranked Results
```

The core insight: faceted search engines already solve the hardest parts of recommendation -- multi-attribute filtering, relevance scoring, typo tolerance, and fast retrieval. Rather than building custom ranking logic from scratch, this design encodes review-derived knowledge into searchable facets and lets the engine's built-in primitives handle retrieval and ranking.

Products are represented as flat documents with typed fields. Each field is either extracted from product metadata (length, weight) or aggregated from review sentiment (stiffness score, edge grip score). The search engine indexes these fields as filterable facets, sortable attributes, and searchable text -- simultaneously.

**Why this works for recommendations:** A recommendation query like "stiff, on-piste carving ski, 180cm+" is structurally identical to a faceted search with filters (`stiffness > 0.7`, `terrain = on-piste`, `length_cm >= 180`) plus a text relevance query (`carving ski`). The user just expresses it in natural language instead of clicking checkboxes.

---

## 2. Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Search engine | **Typesense** (self-hosted via Docker or binary), with **SQLite FTS5 fallback** | Typo-tolerant, fast faceted search, custom ranking, simpler ops than Elasticsearch. Single binary, no JVM. Falls back to SQLite FTS5 + custom faceting when Typesense is unavailable (see Section 10). |
| LLM | **Claude API**, **OpenAI API**, or **Ollama** (local models) | Facet extraction at ingestion; query parsing at search time. Structured JSON output. Provider is swappable via `LLMProvider` abstraction (see Section 10). |
| Orchestration | **Python** scripts + **httpx** for Typesense API | Typesense has a Python client but raw HTTP is more transparent for a POC. |
| Data prep | **SQLite** (staging DB for aggregation) | Intermediate store between raw extraction and final document indexing. Throwaway -- Typesense is the source of truth at runtime. |
| Domain config | **YAML** per domain | Facet definitions, synonym lists, ranking rules. |

**Why Typesense over Meilisearch or Elasticsearch:**
- Typesense supports `sort_by` with custom ranking expressions that can combine multiple numeric fields -- critical for multi-attribute scoring.
- Built-in faceting with counts, ranges, and stats.
- Single binary, ~50MB, no Java runtime.
- Meilisearch is comparable but its ranking rules are less flexible for numeric scoring. Elasticsearch is overkill for local-first POC.

---

## 3. Data Model

### Typesense Collection Schema

Each product domain gets its own Typesense collection. The schema is generated from the domain YAML config.

```json
{
  "name": "skis",
  "fields": [
    {"name": "id",              "type": "string"},
    {"name": "name",            "type": "string",    "sort": true},
    {"name": "brand",           "type": "string",    "facet": true},
    {"name": "domain",          "type": "string",    "facet": true},

    {"name": "terrain",         "type": "string[]",  "facet": true},
    {"name": "ability_level",   "type": "string[]",  "facet": true},

    {"name": "stiffness",       "type": "float",     "facet": true},
    {"name": "edge_grip",       "type": "float",     "facet": true},
    {"name": "stability",       "type": "float",     "facet": true},
    {"name": "playfulness",     "type": "float",     "facet": true},
    {"name": "dampness",        "type": "float",     "facet": true},
    {"name": "weight_feel",     "type": "float",     "facet": true},

    {"name": "length_cm",       "type": "int32",     "facet": true},
    {"name": "waist_mm",        "type": "int32",     "facet": true},
    {"name": "turn_radius_m",   "type": "float",     "facet": true},

    {"name": "has_plate",       "type": "bool",      "facet": true},
    {"name": "has_rocker",      "type": "bool",      "facet": true},

    {"name": "review_count",    "type": "int32"},
    {"name": "avg_sentiment",   "type": "float"},
    {"name": "review_summary",  "type": "string"},

    {"name": "popularity",      "type": "int32"}
  ],
  "default_sorting_field": "popularity"
}
```

**Field types and their roles:**

- **Sentiment-derived floats** (stiffness, edge_grip, etc.): Normalized 0.0--1.0 scores aggregated from review ABSA. These are the core recommendation axes. Typesense can filter (`stiffness:>0.7`) and sort by them.
- **Categorical facets** (terrain, ability_level): Multi-valued string arrays. `terrain: ["on-piste", "all-mountain"]` means the ski is reviewed as suitable for both.
- **Spec fields** (length_cm, waist_mm): Hard product metadata. Support range filters (`length_cm:>=180`).
- **Boolean facets** (has_plate, has_rocker): Enable negation filters ("no plate" maps to `has_plate:false`).
- **review_summary**: A short text synthesis of review consensus. This is the text-search target for soft/vague queries.

---

## 4. Ingestion Pipeline

```
[Raw Reviews] --> [1. Extract] --> [2. Aggregate] --> [3. Build Doc] --> [4. Index]
```

### Step 1: Extract Facets from Reviews

Each review is sent to the LLM for structured facet extraction:

```python
def extract_facets(review_text: str, domain_config: dict) -> dict:
    facet_names = list(domain_config["facets"].keys())
    prompt = f"""Extract product attributes from this review.
    
Attributes to look for: {facet_names}
For numeric sentiment attributes, score 0.0 (very negative) to 1.0 (very positive).
For categorical attributes, pick from the allowed values.
Only include attributes explicitly discussed. Return JSON.

Review: \"{review_text}\""""
    
    response = call_llm(prompt, json_mode=True)
    return validate_against_schema(response, domain_config)
```

Results are staged in SQLite:

```sql
CREATE TABLE extracted_facets (
    review_id   TEXT,
    product_id  TEXT,
    facet_name  TEXT,
    facet_value TEXT,   -- JSON-encoded: number, string, or bool
    confidence  REAL,
    PRIMARY KEY (review_id, facet_name)
);
```

### Step 2: Aggregate Across Reviews

For each product, merge facets from all its reviews:

```python
def aggregate_product_facets(product_id: str, db: sqlite3.Connection) -> dict:
    rows = db.execute(
        "SELECT facet_name, facet_value, confidence FROM extracted_facets "
        "WHERE product_id = ? ORDER BY facet_name", (product_id,)
    ).fetchall()
    
    aggregated = {}
    for name, group in itertools.groupby(rows, key=lambda r: r[0]):
        values = [(json.loads(r[1]), r[2]) for r in group]
        facet_type = domain_config["facets"][name]["type"]
        
        if facet_type == "numeric":
            # Confidence-weighted mean
            total_w = sum(c for _, c in values)
            aggregated[name] = sum(v * c for v, c in values) / total_w
        elif facet_type == "categorical":
            # Union of all mentioned categories
            aggregated[name] = list(set(v for v, _ in values))
        elif facet_type == "boolean":
            # Majority vote
            aggregated[name] = sum(1 for v, _ in values if v) > len(values) / 2
    
    return aggregated
```

### Step 3: Build Typesense Document

Merge aggregated facets with product metadata, generate a review summary text:

```python
def build_document(product_id: str, meta: dict, facets: dict, reviews: list[str]) -> dict:
    summary = generate_review_summary(reviews)  # LLM: 2-sentence consensus
    doc = {
        "id": product_id,
        "name": meta["name"],
        "brand": meta.get("brand", ""),
        "domain": meta["domain"],
        "review_count": len(reviews),
        "avg_sentiment": facets.get("_avg_sentiment", 0.5),
        "review_summary": summary,
        "popularity": meta.get("popularity", 0),
        **facets
    }
    return doc
```

### Step 4: Index in Typesense

```python
def index_products(documents: list[dict], collection_name: str):
    client = typesense.Client({
        "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
        "api_key": "local-dev-key"
    })
    # Upsert handles re-ingestion cleanly
    for doc in documents:
        client.collections[collection_name].documents.upsert(doc)
```

---

## 5. Query Pipeline

```
[NL Query] --> [1. Parse] --> [2. Search] --> [3. Format] --> [Results]
```

### Step 1: LLM Query Parsing

The LLM translates natural language into Typesense search parameters:

```python
def parse_query(nl_query: str, domain_config: dict) -> dict:
    facets = domain_config["facets"]
    prompt = f"""Convert this product query into search parameters.

Query: "{nl_query}"
Available facets: {json.dumps({k: v["type"] for k, v in facets.items()})}

Return JSON with:
- "text_query": words for full-text search (subjective/vague terms)
- "filters": Typesense filter string (e.g., "stiffness:>0.7 && terrain:on-piste")
- "sort_by": optional sort expression (e.g., "stiffness:desc,edge_grip:desc")
- "negations": any "no X" / "without X" mapped to boolean or exclusion filters"""

    result = call_llm(prompt, json_mode=True)
    return validate_and_retry(result, domain_config, nl_query)
```

### Filter Validation and Retry Strategy

LLM-generated filter strings can contain invalid Typesense syntax, referencing nonexistent fields or malformed operators. Every parsed filter is validated before execution, with a simplification-and-retry fallback:

```python
def validate_and_retry(parsed: dict, domain_config: dict, original_query: str) -> dict:
    """Validate LLM-generated filter syntax; retry with simplification on failure."""
    filter_str = parsed.get("filters", "")
    if not filter_str:
        return parsed

    valid_fields = set(domain_config["facets"].keys())
    clauses = [c.strip() for c in filter_str.split("&&")]
    valid_clauses = []
    invalid_clauses = []

    for clause in clauses:
        if _is_valid_clause(clause, valid_fields, domain_config):
            valid_clauses.append(clause)
        else:
            invalid_clauses.append(clause)

    if not invalid_clauses:
        return parsed  # All clauses valid

    if valid_clauses:
        # Drop invalid clauses and proceed with the valid subset
        parsed["filters"] = " && ".join(valid_clauses)
        parsed["_dropped_filters"] = invalid_clauses
        return parsed

    # All clauses invalid -- fall back to text-only search
    parsed["filters"] = ""
    parsed["text_query"] = original_query  # Use the raw NL query as text search
    parsed["_dropped_filters"] = invalid_clauses
    return parsed


def _is_valid_clause(clause: str, valid_fields: set, domain_config: dict) -> bool:
    """Check if a single filter clause references a known field with valid syntax."""
    import re
    # Match patterns like "field_name:>0.7", "field_name:=value", "field_name:>=180"
    match = re.match(r"(\w+)\s*:\s*([><=!]+)?\s*(.+)", clause)
    if not match:
        return False
    field_name = match.group(1)
    if field_name not in valid_fields:
        return False
    facet_def = domain_config["facets"][field_name]
    value = match.group(3).strip()
    # Type-check the value against the facet definition
    if facet_def["type"] in ("numeric", "spec_numeric"):
        try:
            float(value)
        except ValueError:
            return False
    elif facet_def["type"] == "boolean":
        if value.lower() not in ("true", "false"):
            return False
    return True
```

**Example transformations:**

| Natural Language | Parsed Output |
|---|---|
| "stiff, on-piste carving ski, 180cm+" | `filters: "stiffness:>0.7 && terrain:=on-piste && length_cm:>=180"`, `text_query: "carving"` |
| "cushy, responsive running shoe, no plate" | `filters: "cushion:>0.7 && responsiveness:>0.6 && has_plate:false"`, `text_query: "running shoe"` |
| "playful all-mountain ski" | `filters: "terrain:=all-mountain"`, `text_query: "playful"`, `sort_by: "playfulness:desc"` |

### Step 2: Execute Typesense Search

```python
def search(parsed: dict, collection: str) -> list[dict]:
    search_params = {
        "q": parsed.get("text_query", "*"),
        "query_by": "review_summary,name",
        "filter_by": parsed.get("filters", ""),
        "sort_by": parsed.get("sort_by", "_text_match:desc,popularity:desc"),
        "facet_by": "terrain,brand,ability_level",
        "max_hits": 20,
        "per_page": 10
    }
    
    results = client.collections[collection].documents.search(search_params)
    return results["hits"]
```

The search engine handles text relevance, facet filtering, and sorting in a single call. No post-processing pipeline needed for basic queries.

### Step 3: Normalize Scores

Typesense's `text_match_info` scores are engine-internal values (not 0--1). For purely filter-based queries (no text query, i.e., `q: "*"`), text match scores are meaningless. We normalize to a 0--1 scale and fall back to facet-match-quality scoring when appropriate:

```python
def normalize_scores(hits: list[dict], parsed: dict, domain_config: dict) -> list[dict]:
    """Normalize Typesense scores to 0-1 range, with facet-match fallback."""
    is_text_query = parsed.get("text_query", "*") != "*"
    filter_str = parsed.get("filters", "")
    active_filters = _parse_active_filters(filter_str) if filter_str else {}

    if is_text_query and hits:
        # Normalize text_match scores via min-max across the result set
        raw_scores = [
            h.get("text_match_info", {}).get("score", 0) for h in hits
        ]
        max_score = max(raw_scores) if raw_scores else 1
        min_score = min(raw_scores) if raw_scores else 0
        score_range = max_score - min_score if max_score != min_score else 1

        for hit, raw in zip(hits, raw_scores):
            hit["_normalized_score"] = (raw - min_score) / score_range
    else:
        # Filter-only query: score based on how well facet values match filters
        for hit in hits:
            doc = hit["document"]
            hit["_normalized_score"] = _facet_match_score(
                doc, active_filters, domain_config
            )

    return hits


def _facet_match_score(doc: dict, active_filters: dict, domain_config: dict) -> float:
    """Score 0-1 based on how closely facet values satisfy the filter thresholds."""
    if not active_filters:
        return 0.5  # No filters, neutral score

    scores = []
    for field, (op, target) in active_filters.items():
        val = doc.get(field)
        if val is None:
            scores.append(0.0)
            continue
        facet_type = domain_config["facets"].get(field, {}).get("type", "")
        if facet_type in ("numeric", "spec_numeric"):
            target_f = float(target)
            val_f = float(val)
            if op in (">", ">="):
                # How far above the threshold? Clamp to [0, 1]
                scores.append(min(1.0, max(0.0, val_f / target_f if target_f else 1.0)))
            elif op in ("<", "<="):
                scores.append(min(1.0, max(0.0, target_f / val_f if val_f else 1.0)))
            else:
                scores.append(1.0 if val_f == target_f else 0.0)
        elif facet_type == "categorical":
            if isinstance(val, list):
                scores.append(1.0 if target in val else 0.0)
            else:
                scores.append(1.0 if val == target else 0.0)
        elif facet_type == "boolean":
            scores.append(1.0 if str(val).lower() == target.lower() else 0.0)
        else:
            scores.append(0.5)

    return sum(scores) / len(scores) if scores else 0.5


def _parse_active_filters(filter_str: str) -> dict[str, tuple[str, str]]:
    """Parse 'field:>0.7 && field2:=value' into {field: (op, value)}."""
    import re
    filters = {}
    for clause in filter_str.split("&&"):
        clause = clause.strip()
        match = re.match(r"(\w+)\s*:\s*([><=!]+)?\s*(.+)", clause)
        if match:
            filters[match.group(1)] = (match.group(2) or "=", match.group(3).strip())
    return filters
```

### Step 4: Format Results with Explanations

Build `matched_attributes` from the document's facet values. For each facet that was part of the filter, include the product's actual value and how well it satisfied the filter threshold:

```python
def format_results(
    hits: list[dict], parsed: dict, domain_config: dict
) -> list[RecommendationResult]:
    filter_str = parsed.get("filters", "")
    active_filters = _parse_active_filters(filter_str) if filter_str else {}
    results = []

    for hit in hits:
        doc = hit["document"]
        matched_attributes = {}

        for field, (op, target) in active_filters.items():
            val = doc.get(field)
            if val is None:
                continue
            facet_type = domain_config["facets"].get(field, {}).get("type", "")
            if facet_type in ("numeric", "spec_numeric"):
                target_f = float(target)
                val_f = float(val)
                if op in (">", ">="):
                    match_quality = min(1.0, max(0.0, val_f / target_f if target_f else 1.0))
                elif op in ("<", "<="):
                    match_quality = min(1.0, max(0.0, target_f / val_f if val_f else 1.0))
                else:
                    match_quality = 1.0 if val_f == target_f else 0.0
                matched_attributes[field] = match_quality
            elif facet_type == "categorical":
                matched_attributes[field] = 1.0 if (
                    target in val if isinstance(val, list) else target == val
                ) else 0.0
            elif facet_type == "boolean":
                matched_attributes[field] = 1.0 if str(val).lower() == target.lower() else 0.0

        # Build human-readable explanation
        explanation_parts = []
        for field, quality in matched_attributes.items():
            actual = doc.get(field)
            if isinstance(actual, float):
                explanation_parts.append(f"{field}={actual:.2f} (match: {quality:.0%})")
            else:
                explanation_parts.append(f"{field}={actual} (match: {quality:.0%})")
        explanation = "; ".join(explanation_parts) if explanation_parts else doc.get("review_summary", "")[:120]

        results.append(RecommendationResult(
            product_id=doc["id"],
            product_name=doc["name"],
            score=hit.get("_normalized_score", 0.0),
            explanation=explanation,
            matched_attributes=matched_attributes,
        ))

    return results
```

---

## 6. Ranking Strategy

Typesense's ranking is configured through `sort_by` expressions and custom ranking rules, not application-side code.

### Default Ranking

```
sort_by: "_text_match:desc, popularity:desc"
```

Text match score handles queries with subjective terms. Popularity breaks ties.

### Query-Adaptive Ranking

When the LLM parser identifies dominant facets, it generates a custom `sort_by`:

```
# "stiff on-piste ski" -> user cares about stiffness + edge grip
sort_by: "_text_match:desc, stiffness:desc, edge_grip:desc, popularity:desc"
```

Typesense evaluates these left-to-right as tiebreakers, which naturally prioritizes text relevance, then the user's stated preferences, then general popularity.

### Boosting via Overrides

Typesense supports "overrides" (pinning/boosting rules) that can be configured per domain:

```python
# Boost products with high review counts (more trustworthy scores)
override = {
    "id": "boost-well-reviewed",
    "rule": {"query": "*", "match": "exact"},
    "sort_by": "review_count:desc"
}
```

### What the Engine Gives Us for Free

- **Typo tolerance**: "stif on-pist ski" still matches.
- **Prefix search**: partial terms work during interactive input.
- **Facet counts**: response includes `{"terrain": [{"value": "on-piste", "count": 12}]}` -- useful for "did you mean?" style refinement.
- **Geolocation** (future): Typesense supports geo fields if location relevance matters.

---

## 7. Domain Adaptation

Each domain is defined by a YAML file that drives schema generation, extraction prompts, and query parsing:

```yaml
# domains/ski.yaml
domain: ski
collection: skis
facets:
  stiffness:
    type: numeric
    range: [0.0, 1.0]
    description: "Flex pattern from soft/playful to stiff/demanding"
    synonyms: ["flex", "stiff", "soft", "rigid"]
  edge_grip:
    type: numeric
    range: [0.0, 1.0]
    description: "Hold on hard snow and ice"
    synonyms: ["edge hold", "grip", "ice performance"]
  terrain:
    type: categorical
    values: ["on-piste", "off-piste", "all-mountain", "park", "touring"]
    synonyms: ["groomer", "powder", "backcountry", "resort"]
  length_cm:
    type: spec_numeric
    unit: cm
    description: "Ski length"
  has_rocker:
    type: boolean
    description: "Whether the ski has rocker profile"
    synonyms: ["rocker", "camber"]

ranking_defaults:
  sort_by: "_text_match:desc,popularity:desc"
  primary_facets: ["stiffness", "edge_grip", "terrain"]
```

### Adding a New Domain

1. Write the YAML config file.
2. Run `python manage.py create-collection --domain running_shoe` to generate and create the Typesense schema.
3. Ingest reviews. The extraction LLM prompt is built dynamically from the facet definitions.
4. Query. The parser LLM prompt is built dynamically from the same config.

No code changes required. The YAML drives everything.

```python
def build_typesense_schema(domain_config: dict) -> dict:
    fields = [
        {"name": "id",   "type": "string"},
        {"name": "name", "type": "string", "sort": True},
    ]
    for facet_name, facet_def in domain_config["facets"].items():
        ts_type = {
            "numeric": "float",
            "categorical": "string[]",
            "spec_numeric": "int32",
            "boolean": "bool"
        }[facet_def["type"]]
        fields.append({"name": facet_name, "type": ts_type, "facet": True})
    
    fields += [
        {"name": "review_summary", "type": "string"},
        {"name": "review_count",   "type": "int32"},
        {"name": "popularity",     "type": "int32"},
    ]
    return {"name": domain_config["collection"], "fields": fields}
```

---

## 8. Pros and Cons

### Strengths

- **Minimal custom code.** The search engine handles indexing, filtering, ranking, typo tolerance, and faceting. The application layer is thin -- just LLM-powered extraction and query parsing.
- **Fast queries.** Typesense is optimized for sub-50ms search latency. Once the query is parsed (one LLM call), retrieval and ranking are near-instant. Total query time is dominated by LLM parsing (~1-2s), not search.
- **Battle-tested infrastructure.** Faceted search is a solved problem. Typesense/Elasticsearch have years of production hardening. We inherit their reliability.
- **Built-in facet exploration.** Facet counts in responses enable "refine your search" UIs naturally. "12 on-piste skis, 8 all-mountain" helps users narrow results interactively.
- **Operationally simple.** Typesense is a single binary. No vector index to maintain separately, no dual-store synchronization bugs. One data store, one query path.

### Weaknesses

- **No semantic understanding in retrieval.** Text matching on `review_summary` is keyword-based, not semantic. "Butter-smooth ride" will not match a query for "damp vibration absorption" unless the summary happens to use similar words. This is the fundamental limitation.
- **LLM query parsing is fragile (mitigated).** The system depends on the LLM correctly translating NL to Typesense filter syntax. Invalid filter clauses are now validated and dropped with a simplification-and-retry strategy (see Section 5), falling back to text-only search in the worst case. Vague queries ("good beginner ski") remain hard to express as facet filters.
- **Flat document model.** Faceted search assumes products can be described by a fixed set of typed fields. Products with unusual or novel attributes (a ski with a unique dampening technology) cannot be represented unless the schema is extended.
- **Aggregation loses nuance.** Collapsing 20 reviews into a single `stiffness: 0.73` score discards the distribution. If 10 reviewers say "stiff" and 10 say "medium," the average masks disagreement. No way to surface "reviewers disagree on flex" within the search engine model.
- **No cross-product reasoning.** The search engine scores each product independently. It cannot reason about diversity in results ("you already have two on-piste recommendations, here is an all-mountain alternative") or comparative features.
- **External dependency (mitigated).** Typesense must be running as a process for full functionality. A SQLite FTS5 fallback is provided for environments where Docker is unavailable, but it lacks typo tolerance and advanced ranking. The fallback is sufficient for benchmarking correctness but not representative of production search quality.

---

## 9. POC Scope

### Goal
Demonstrate end-to-end flow: ingest ski reviews, index as faceted documents, answer NL queries with facet-filtered ranked results.

### File Structure

```
recommend/
  manage.py                # CLI: create-collection, ingest, search
  search/
    interface.py           # Recommender protocol + RecommendationResult
    llm_provider.py        # LLM abstraction (Ollama / Claude API / OpenAI)
    backend.py             # Search backend abstraction (Typesense / SQLite FTS5)
    extraction.py          # LLM facet extraction from reviews
    aggregation.py         # Per-product facet aggregation
    indexer.py             # Typesense document creation + upsert
    query_parser.py        # NL -> Typesense search params (LLM) + filter validation
    scoring.py             # Score normalization + facet-match fallback scoring
    searcher.py            # Execute search, format results
  domains/
    ski.yaml
  data/
    sample_reviews.json
  docker-compose.yml       # Typesense server
```

### Infrastructure Setup

**Option A: Typesense via Docker (full-featured)**

The benchmark runner should start Typesense before running Design 9. The `docker-compose.yml` is self-contained:

```bash
# Start Typesense (run once, persists across benchmark runs)
cd recommend/
docker compose up -d typesense

# Verify it's running
curl http://localhost:8108/health  # should return {"ok": true}

# Tear down when done
docker compose down
```

```yaml
# docker-compose.yml
services:
  typesense:
    image: typesense/typesense:27.1
    ports:
      - "8108:8108"
    volumes:
      - typesense-data:/data
    command: "--data-dir /data --api-key=local-dev-key"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8108/health"]
      interval: 5s
      timeout: 3s
      retries: 5
volumes:
  typesense-data:
```

**Option B: SQLite FTS5 fallback (zero external dependencies)**

When Typesense is not available (no Docker, CI environments, quick local testing), the system falls back to SQLite FTS5 with custom faceting logic. This provides the same API surface with reduced search quality (no typo tolerance, simpler ranking):

```python
class SQLiteFallbackBackend:
    """Drop-in replacement for Typesense backend using SQLite FTS5."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")

    def create_collection(self, schema: dict):
        fields = schema["fields"]
        cols = ", ".join(f"{f['name']} {self._sqlite_type(f['type'])}" for f in fields)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {schema['name']} ({cols})")
        # FTS5 virtual table for text search
        text_fields = [f["name"] for f in fields if f["type"] == "string" and f["name"] != "id"]
        if text_fields:
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {schema['name']}_fts "
                f"USING fts5({', '.join(text_fields)}, content={schema['name']})"
            )

    def search(self, collection: str, params: dict) -> list[dict]:
        query = params.get("q", "*")
        filter_by = params.get("filter_by", "")

        if query != "*":
            # Use FTS5 for text search
            sql = f"SELECT * FROM {collection}_fts WHERE {collection}_fts MATCH ?"
            rows = self.conn.execute(sql, (query,)).fetchall()
        else:
            sql = f"SELECT * FROM {collection}"
            where_clauses = self._translate_filters(filter_by)
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            rows = self.conn.execute(sql).fetchall()

        return rows

    @staticmethod
    def _sqlite_type(ts_type: str) -> str:
        return {"string": "TEXT", "string[]": "TEXT", "float": "REAL",
                "int32": "INTEGER", "bool": "INTEGER"}.get(ts_type, "TEXT")

    @staticmethod
    def _translate_filters(filter_str: str) -> list[str]:
        """Convert Typesense filter syntax to SQL WHERE clauses."""
        if not filter_str:
            return []
        import re
        clauses = []
        for part in filter_str.split("&&"):
            part = part.strip()
            m = re.match(r"(\w+)\s*:\s*([><=!]+)?\s*(.+)", part)
            if m:
                field, op, val = m.group(1), m.group(2) or "=", m.group(3).strip()
                sql_op = {"=": "=", ">": ">", "<": "<", ">=": ">=", "<=": "<="}.get(op, "=")
                clauses.append(f"{field} {sql_op} {val}")
        return clauses
```

Backend selection is automatic:

```python
def _detect_backend() -> str:
    """Check if Typesense is available; fall back to SQLite FTS5."""
    try:
        import httpx
        resp = httpx.get("http://localhost:8108/health", timeout=2.0)
        if resp.status_code == 200:
            return "typesense"
    except Exception:
        pass
    return "sqlite"
```

### Code Sketch: Query Flow

```python
# manage.py (simplified)
import typer
import httpx
import json
from search.query_parser import parse_nl_query
from search.searcher import execute_search

TYPESENSE_URL = "http://localhost:8108"
HEADERS = {"X-TYPESENSE-API-KEY": "local-dev-key"}
app = typer.Typer()

@app.command()
def search(query: str, domain: str = "ski"):
    # Step 1: LLM parses NL into search params
    domain_config = load_domain_config(domain)
    parsed = parse_nl_query(query, domain_config)
    print(f"Parsed: filters={parsed['filters']}, text={parsed['text_query']}")
    
    # Step 2: Execute against Typesense
    search_params = {
        "q": parsed.get("text_query", "*"),
        "query_by": "review_summary,name",
        "filter_by": parsed.get("filters", ""),
        "sort_by": parsed.get("sort_by", "_text_match:desc,popularity:desc"),
        "per_page": 10,
        "facet_by": ",".join(
            f for f, d in domain_config["facets"].items()
            if d["type"] == "categorical"
        )
    }
    
    resp = httpx.get(
        f"{TYPESENSE_URL}/collections/{domain_config['collection']}/documents/search",
        params=search_params, headers=HEADERS
    )
    results = resp.json()
    
    # Step 3: Display
    print(f"\nFound {results['found']} results:\n")
    for i, hit in enumerate(results["hits"], 1):
        doc = hit["document"]
        print(f"  {i}. {doc['name']}")
        print(f"     Stiffness: {doc.get('stiffness', 'N/A'):.2f}  "
              f"Edge grip: {doc.get('edge_grip', 'N/A'):.2f}  "
              f"Terrain: {doc.get('terrain', [])}")
        print(f"     Summary: {doc.get('review_summary', '')[:120]}")
        print()
    
    # Show facet breakdown
    if results.get("facet_counts"):
        print("Facet breakdown:")
        for facet in results["facet_counts"]:
            vals = ", ".join(f"{c['value']}({c['count']})" for c in facet["counts"][:5])
            print(f"  {facet['field_name']}: {vals}")

@app.command()
def ingest(file: str, domain: str = "ski"):
    from search.extraction import extract_facets
    from search.aggregation import aggregate_product_facets
    from search.indexer import index_documents
    
    domain_config = load_domain_config(domain)
    reviews = json.loads(open(file).read())
    
    # Extract -> Aggregate -> Index
    extracted = {}
    for review in reviews:
        facets = extract_facets(review["text"], domain_config)
        extracted.setdefault(review["product_id"], []).append(facets)
    
    documents = []
    for product_id, facet_list in extracted.items():
        aggregated = aggregate_product_facets(facet_list, domain_config)
        doc = {"id": product_id, **aggregated}
        documents.append(doc)
    
    index_documents(documents, domain_config["collection"])
    print(f"Indexed {len(documents)} products")

if __name__ == "__main__":
    app()
```

### POC Milestones

1. **Typesense running locally** via Docker. Create ski collection from YAML config.
2. **Ingest 20-30 ski reviews** covering 10-15 products. Verify facets are extracted and aggregated correctly.
3. **Run 5 test queries** spanning the difficulty spectrum:
   - Filter-heavy: "stiff on-piste carving ski, 180cm+"
   - Negation: "all-mountain ski, no rocker"
   - Vague/subjective: "playful, fun ski for a good skier"
   - Mixed: "damp, stable ski under 90mm waist"
   - Exploratory: "best ski for ice coast"
4. **Evaluate:** Check whether LLM query parsing produces correct Typesense filters. Check whether facet scores meaningfully differentiate products. Identify queries where the approach breaks down.

### Dependencies

```
httpx>=0.24.0            # Typesense HTTP API + Ollama API
typer>=0.9.0             # CLI framework
pyyaml>=6.0              # Domain configs

# LLM providers (install at least one):
anthropic>=0.30.0        # Claude API
openai>=1.0.0            # OpenAI API (also works with compatible endpoints)
# Ollama requires no pip package -- just a running Ollama server
```

Plus Typesense via Docker (optional -- falls back to SQLite FTS5).

---

## 10. Common Interface

All 10 benchmark designs share the same `Recommender` protocol and `RecommendationResult` dataclass. This design's implementation wraps Typesense (or the SQLite FTS5 fallback) behind this interface so the benchmark runner can treat every design identically.

### Protocol and Data Types

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

### Implementation: FacetedSearchRecommender

```python
class FacetedSearchRecommender:
    """Wraps Typesense (or SQLite FTS5 fallback) behind the common Recommender interface."""

    def __init__(self, llm: "LLMProvider | None" = None):
        self.llm = llm or get_default_llm_provider()
        self.backend_type = _detect_backend()
        if self.backend_type == "typesense":
            self.client = typesense.Client({
                "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
                "api_key": "local-dev-key",
            })
        else:
            self.fallback = SQLiteFallbackBackend()
        self.domain_configs: dict[str, dict] = {}

    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        domain_config = load_domain_config(domain)
        self.domain_configs[domain] = domain_config

        # Step 1: Extract facets from reviews via LLM
        extracted: dict[str, list[dict]] = {}
        for review in reviews:
            facets = extract_facets(review["text"], domain_config, self.llm)
            extracted.setdefault(review["product_id"], []).append(facets)

        # Step 2: Aggregate per product
        documents = []
        product_lookup = {p["id"]: p for p in products}
        for product_id, facet_list in extracted.items():
            meta = product_lookup.get(product_id, {"id": product_id, "name": product_id, "domain": domain})
            aggregated = aggregate_product_facets(facet_list, domain_config)
            doc = build_document(product_id, meta, aggregated, [
                r["text"] for r in reviews if r["product_id"] == product_id
            ])
            documents.append(doc)

        # Step 3: Create collection + index
        schema = build_typesense_schema(domain_config)
        if self.backend_type == "typesense":
            try:
                self.client.collections[schema["name"]].delete()
            except Exception:
                pass
            self.client.collections.create(schema)
            for doc in documents:
                self.client.collections[schema["name"]].documents.upsert(doc)
        else:
            self.fallback.create_collection(schema)
            for doc in documents:
                self.fallback.upsert(schema["name"], doc)

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        domain_config = self.domain_configs.get(domain) or load_domain_config(domain)

        # Step 1: LLM parses NL -> search params (with validation + retry)
        parsed = parse_query(query_text, domain_config, self.llm)

        # Step 2: Execute search
        search_params = {
            "q": parsed.get("text_query", "*"),
            "query_by": "review_summary,name",
            "filter_by": parsed.get("filters", ""),
            "sort_by": parsed.get("sort_by", "_text_match:desc,popularity:desc"),
            "per_page": top_k,
        }

        if self.backend_type == "typesense":
            results = self.client.collections[domain_config["collection"]].documents.search(search_params)
            hits = results["hits"]
        else:
            hits = self.fallback.search(domain_config["collection"], search_params)

        # Step 3: Normalize scores
        hits = normalize_scores(hits, parsed, domain_config)

        # Step 4: Format into RecommendationResult
        return format_results(hits, parsed, domain_config)
```

### LLM Provider Abstraction

The system supports both local (Ollama) and remote API providers. The LLM is used in two places -- facet extraction during ingestion and query parsing at search time -- and the provider is swappable without changing any other code:

```python
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstraction over LLM providers for structured JSON output."""

    @abstractmethod
    def complete(self, prompt: str, json_mode: bool = False) -> str:
        """Send a prompt and return the response text."""
        ...


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama REST API."""

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        import httpx
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
        resp = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=60.0)
        return resp.json()["response"]


class AnthropicProvider(LLMProvider):
    """Claude API via the anthropic SDK."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        system = "Respond with valid JSON only." if json_mode else ""
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API provider."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.model = model

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content


def get_default_llm_provider() -> LLMProvider:
    """Auto-detect available LLM provider, preferring local Ollama."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            return OllamaProvider()
    except Exception:
        pass
    try:
        import anthropic
        anthropic.Anthropic()  # Checks ANTHROPIC_API_KEY
        return AnthropicProvider()
    except Exception:
        pass
    try:
        from openai import OpenAI
        OpenAI()  # Checks OPENAI_API_KEY
        return OpenAIProvider()
    except Exception:
        pass
    raise RuntimeError(
        "No LLM provider available. Install Ollama locally, "
        "or set ANTHROPIC_API_KEY or OPENAI_API_KEY."
    )
```

The `call_llm(prompt, json_mode)` helper used throughout earlier sections delegates to whichever `LLMProvider` is active:

```python
def call_llm(prompt: str, json_mode: bool = False, llm: LLMProvider | None = None) -> dict:
    provider = llm or get_default_llm_provider()
    response_text = provider.complete(prompt, json_mode=json_mode)
    return json.loads(response_text)
```

---

## Summary

This design makes a deliberate bet: that the gap between "faceted search" and "recommendation" is smaller than it appears, and that an LLM can bridge it by translating both directions -- reviews into facets, queries into filters. The search engine is not a compromise; it is the right tool for structured multi-attribute retrieval with known facets.

The design is strongest when queries map cleanly to facet filters ("stiff, on-piste, 180cm+"). It is weakest when queries are vague, subjective, or require semantic understanding that keyword search cannot provide ("something that feels alive underfoot"). The honest question for evaluation is whether the strong cases are common enough to justify the simplicity tradeoff -- and whether the weak cases can be patched with better review summaries rather than a fundamentally different architecture.
