# Recommendation System Benchmark

This benchmark evaluates 10 competing recommendation system implementations against a shared test dataset with ground-truth rankings. The goal is to determine which design produces the most relevant, well-ranked, and well-explained product recommendations from natural language queries.

All 10 implementations share the same interface:

```python
class Recommender(Protocol):
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None: ...
    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]: ...

@dataclass
class RecommendationResult:
    product_id: str
    product_name: str
    score: float  # 0-1 normalized
    explanation: str
    matched_attributes: dict[str, float]
```

---

## 1. Test Dataset

### Ski Domain (Primary)

**File:** `data/ski_products.json` — 25 ski products  
**File:** `data/ski_reviews.json` — 135 reviews (5-6 per product)

The ski products are deliberately differentiated across a spectrum of categories:

| Category | Products | Key Traits |
|----------|----------|------------|
| Race (slalom/GS) | SKI-001, SKI-002 | Stiffness 9-10, edge grip 10, zero forgiveness, camber-only, narrow (<70mm) |
| Expert carving | SKI-004, SKI-023 | Stiffness 8, dampness 7-9, edge grip 9, camber, 72mm waist |
| Advanced frontside | SKI-005, SKI-015 | Stiffness 7, edge grip 8, on-piste focus, 76-82mm waist |
| All-mountain frontside | SKI-021, SKI-025 | Stiffness 7, edge grip 8, camber + tip rocker, 88-90mm waist |
| All-mountain | SKI-006, SKI-014, SKI-019, SKI-020 | 96-100mm waist, mixed rocker profiles, balanced attributes |
| Freeride | SKI-008, SKI-009, SKI-010, SKI-013 | 106-108mm, rocker-camber-rocker, stiffness varies 6-8 |
| Powder/freeride | SKI-011, SKI-012 | 108-112mm, full/aggressive rocker, float 9-10 |
| Freestyle/park | SKI-018 | Playfulness 9, twin tip, park terrain |
| All-mountain freeride | SKI-007, SKI-024 | Playful, 102-106mm, versatile |
| Big mountain | SKI-022 | Stiff, damp, heavy, stability-focused |
| Beginner | SKI-003, SKI-016, SKI-017 | Stiffness 2-4, forgiveness 8-10, lightweight |

Each product has:
- **Specs:** lengths available (cm), waist width (mm), turn radius (m), weight (g/ski), rocker profile, construction type, binding system
- **Attributes** (1-10 scale): stiffness, dampness, edge grip, stability at speed, playfulness, powder float, forgiveness
- **Terrain tags:** on-piste, off-piste, all-mountain, freeride, powder, park, race, etc.

Reviews are written to sound realistic and cover the attribute space with specific language (e.g., "feels alive underfoot", "ice coast", "confidence-inspiring"). Some reviews deliberately use exact phrases that appear in test queries, testing whether systems can do semantic matching vs. keyword matching.

### Running Shoe Domain (Cross-Domain Validation)

**File:** `data/running_shoes.json` — 10 products, 50 reviews

Covers road runners (cushioned, racing, daily) and trail runners (aggressive grip, versatile). Used by the two cross-domain queries to verify that systems can adapt between domains.

---

## 2. Test Queries

**File:** `data/test_queries.json` — 20 queries across 5 difficulty levels

### Easy (5 queries) — Single clear attribute
| ID | Query | Tests |
|----|-------|-------|
| EASY-01 | "stiff carving ski" | Basic attribute matching (stiffness + edge grip) |
| EASY-02 | "soft beginner ski" | Inverse attribute matching (low stiffness + high forgiveness) |
| EASY-03 | "powder ski with good float" | Direct attribute lookup (powder float) |
| EASY-04 | "playful freestyle ski for the park" | Attribute + terrain matching |
| EASY-05 | "damp stable all-mountain ski" | Multi-attribute matching within one category |

### Medium (5 queries) — Multiple constraints
| ID | Query | Tests |
|----|-------|-------|
| MED-01 | "stiff, damp, on-piste ski available in 180cm or longer" | Attribute + spec constraint (length) |
| MED-02 | "lightweight freeride ski with good playfulness and powder float" | Attribute + spec (weight) + category |
| MED-03 | "titanal construction ski with edge grip for hardpack and at least 95mm waist" | Construction material + attribute + spec |
| MED-04 | "versatile one-ski quiver around 96-100mm waist for mixed conditions" | Concept ("quiver of one") + spec range |
| MED-05 | "expert carving ski with dampness above 7 and full camber profile" | Attribute threshold + rocker profile |

### Hard (5 queries) — Negations, ranges, trade-offs
| ID | Query | Tests |
|----|-------|-------|
| HARD-01 | "all-mountain ski with no rocker, under 90mm waist" | Negation (no rocker) + range constraint |
| HARD-02 | "freeride ski that is NOT playful, stiff with high stability, over 105mm waist" | Negation (NOT playful) + multiple attribute thresholds |
| HARD-03 | "ski for advanced intermediate that's forgiving but not a beginner ski, 80-95mm waist, available in 170-175cm" | Skill-level nuance + multiple range constraints |
| HARD-04 | "the lightest ski that still has good edge grip and stability for groomed runs" | Optimization (minimize weight, subject to constraints) |
| HARD-05 | "ski that handles both powder and hardpack equally well, not a specialist in either" | Balance/trade-off reasoning |

### Vague (3 queries) — Subjective/metaphorical language
| ID | Query | Tests |
|----|-------|-------|
| VAGUE-01 | "a ski that feels alive underfoot" | Metaphorical language (review text matching) |
| VAGUE-02 | "good ski for the ice coast" | Slang/regional term interpretation |
| VAGUE-03 | "a confidence-inspiring ski for someone who skis fast but wants to feel safe" | Subjective multi-attribute reasoning |

### Cross-Domain (2 queries) — Running shoes
| ID | Query | Tests |
|----|-------|-------|
| XDOM-01 | "cushioned long-distance running shoe for marathon training" | Domain switching, attribute mapping |
| XDOM-02 | "lightweight responsive trail running shoe with good grip" | Domain switching, multi-attribute in new domain |

---

## 3. Ground Truth and Scoring

Each query has a `ground_truth_top5` with manually curated relevance grades:

- **3 = Highly relevant** — This product is an excellent match for the query
- **2 = Relevant** — This product is a good match with minor misalignment
- **1 = Marginally relevant** — This product is partially relevant but not ideal

Products not in the ground truth are scored as **0 = Not relevant**.

These grades enable NDCG (Normalized Discounted Cumulative Gain) computation using standard IR methodology.

---

## 4. Evaluation Metrics

### 4.1 Ranking Quality

**NDCG@k (k=5, k=10)**

Measures how well the system ranks results compared to the ideal ordering. Uses the standard formula:

```
DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

Where `rel_i` is the relevance grade (0-3) of the item at position i, and IDCG is the DCG of the ideal ranking.

**Mean Reciprocal Rank (MRR)**

```
MRR = (1/|Q|) * sum_{q in Q} 1/rank_q
```

Where `rank_q` is the position of the first highly relevant (grade 3) result for query q.

### 4.2 Attribute Precision

For each result, the system returns `matched_attributes: dict[str, float]`. We evaluate:

```
Attribute Precision = |predicted_attributes ∩ expected_attributes| / |predicted_attributes|
Attribute Recall = |predicted_attributes ∩ expected_attributes| / |expected_attributes|
Attribute F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Where `expected_attributes` are the `key_attributes` defined for each query.

### 4.3 Query Coverage

```
Coverage = |{q : q has at least 1 result with relevance >= 2}| / |Q|
```

The percentage of queries where the system returns at least one relevant result.

### 4.4 Latency

- **Ingestion latency:** Total time for `ingest()` across all domains (ms)
- **Query latency (p50, p95, p99):** Per-query time for `query()` (ms)
- Measured over 3 runs, median reported

### 4.5 Explanation Quality

Evaluated on a 1-5 rubric per result, averaged across all results:

| Score | Criteria |
|-------|----------|
| 5 | Explanation references specific product attributes AND review content, is factually grounded |
| 4 | References product attributes correctly, readable |
| 3 | Generic but accurate explanation |
| 2 | Vague or partially inaccurate |
| 1 | Wrong, hallucinated, or empty |

Automated proxy: check if the explanation mentions at least one attribute name that appears in the product's actual attributes. Full human evaluation on a random sample of 50 results.

---

## 5. Benchmark Runner

### Directory Structure

```
benchmark/
├── README.md                    # This file
├── data/
│   ├── ski_products.json        # 25 ski products
│   ├── ski_reviews.json         # 135 ski reviews
│   ├── running_shoes.json       # 10 shoe products + 50 reviews
│   └── test_queries.json        # 20 test queries with ground truth
├── runner.py                    # Main benchmark runner
├── metrics.py                   # Metric computation
├── report.py                    # Report generation
└── results/                     # Output directory (gitignored)
    ├── summary.json
    ├── summary.txt
    └── per_query/
        └── {recommender}_{query_id}.json
```

### Runner Implementation

```python
"""
benchmark/runner.py — Main benchmark runner

Usage:
    python runner.py --recommenders path/to/designs/ --output results/
    python runner.py --recommenders path/to/designs/ --filter "design_01,design_05"
"""

import argparse
import importlib
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Protocol

from metrics import (
    compute_ndcg,
    compute_mrr,
    compute_attribute_precision,
    compute_coverage,
    compute_explanation_quality_proxy,
)
from report import generate_report


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Path) -> dict:
    """Load all benchmark data files."""
    with open(data_dir / "ski_products.json") as f:
        ski_products = json.load(f)
    with open(data_dir / "ski_reviews.json") as f:
        ski_reviews = json.load(f)
    with open(data_dir / "running_shoes.json") as f:
        shoes = json.load(f)
    with open(data_dir / "test_queries.json") as f:
        queries = json.load(f)

    return {
        "domains": {
            "ski": {
                "products": ski_products["products"],
                "reviews": ski_reviews["reviews"],
            },
            "running_shoe": {
                "products": shoes["products"],
                "reviews": shoes["reviews"],
            },
        },
        "queries": queries["queries"],
    }


# ---------------------------------------------------------------------------
# Recommender discovery
# ---------------------------------------------------------------------------

def discover_recommenders(designs_dir: Path, filter_names: list[str] | None = None):
    """
    Load recommender implementations from the designs directory.

    Each design is expected to be a Python module or package with a
    `create_recommender() -> Recommender` factory function.

    Expected layout:
        designs/
        ├── design_01/
        │   └── __init__.py   # contains create_recommender()
        ├── design_02/
        │   └── __init__.py
        ...
    """
    recommenders = {}
    sys.path.insert(0, str(designs_dir.parent))

    for entry in sorted(designs_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        if filter_names and entry.name not in filter_names:
            continue
        try:
            module = importlib.import_module(f"{designs_dir.name}.{entry.name}")
            recommender = module.create_recommender()
            recommenders[entry.name] = recommender
            print(f"  Loaded: {entry.name}")
        except Exception as e:
            print(f"  FAILED to load {entry.name}: {e}")

    return recommenders


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query_id: str
    query_text: str
    difficulty: str
    domain: str
    results: list[dict]          # raw RecommendationResult dicts
    latency_ms: float
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    attribute_precision: float
    attribute_recall: float
    attribute_f1: float
    explanation_score: float
    ground_truth: list[dict]


@dataclass
class RecommenderBenchmark:
    name: str
    ingestion_latency_ms: float
    query_results: list[QueryResult]
    # Aggregates (filled after all queries)
    mean_ndcg_5: float = 0.0
    mean_ndcg_10: float = 0.0
    mean_mrr: float = 0.0
    mean_attr_f1: float = 0.0
    coverage: float = 0.0
    mean_explanation: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0


def run_benchmark(
    recommenders: dict,
    dataset: dict,
    num_runs: int = 3,
) -> list[RecommenderBenchmark]:
    """Run the full benchmark suite for all recommenders."""

    queries = dataset["queries"]
    results = []

    for name, recommender in recommenders.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")

        # ----- Ingestion -----
        ingestion_times = []
        for run in range(num_runs):
            t0 = time.perf_counter()
            for domain_name, domain_data in dataset["domains"].items():
                recommender.ingest(
                    products=domain_data["products"],
                    reviews=domain_data["reviews"],
                    domain=domain_name,
                )
            t1 = time.perf_counter()
            ingestion_times.append((t1 - t0) * 1000)

        ingestion_ms = sorted(ingestion_times)[len(ingestion_times) // 2]
        print(f"  Ingestion: {ingestion_ms:.1f} ms (median of {num_runs} runs)")

        # ----- Queries -----
        query_results = []
        for query in queries:
            latencies = []
            all_run_results = []

            for run in range(num_runs):
                t0 = time.perf_counter()
                rec_results = recommender.query(
                    query_text=query["query_text"],
                    domain=query["domain"],
                    top_k=10,
                )
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                all_run_results.append(rec_results)

            # Use median-latency run's results
            median_idx = sorted(range(len(latencies)), key=lambda i: latencies[i])[
                len(latencies) // 2
            ]
            rec_results = all_run_results[median_idx]
            latency_ms = latencies[median_idx]

            # Convert results to dicts for serialization
            result_dicts = []
            for r in rec_results:
                result_dicts.append({
                    "product_id": r.product_id,
                    "product_name": r.product_name,
                    "score": r.score,
                    "explanation": r.explanation,
                    "matched_attributes": r.matched_attributes,
                })

            # Build ground truth relevance map
            gt_relevance = {
                item["product_id"]: item["relevance"]
                for item in query["ground_truth_top5"]
            }

            # Compute per-query metrics
            predicted_ids = [r["product_id"] for r in result_dicts]
            ndcg_5 = compute_ndcg(predicted_ids, gt_relevance, k=5)
            ndcg_10 = compute_ndcg(predicted_ids, gt_relevance, k=10)
            mrr = compute_mrr(predicted_ids, gt_relevance, threshold=3)

            attr_p, attr_r, attr_f1 = compute_attribute_precision(
                result_dicts, query.get("key_attributes", [])
            )
            expl_score = compute_explanation_quality_proxy(
                result_dicts,
                dataset["domains"].get(query["domain"], {}).get("products", []),
            )

            query_results.append(QueryResult(
                query_id=query["id"],
                query_text=query["query_text"],
                difficulty=query["difficulty"],
                domain=query["domain"],
                results=result_dicts,
                latency_ms=latency_ms,
                ndcg_at_5=ndcg_5,
                ndcg_at_10=ndcg_10,
                mrr=mrr,
                attribute_precision=attr_p,
                attribute_recall=attr_r,
                attribute_f1=attr_f1,
                explanation_score=expl_score,
                ground_truth=query["ground_truth_top5"],
            ))

            status = "OK" if ndcg_5 > 0.5 else "WEAK" if ndcg_5 > 0 else "MISS"
            print(f"  [{status}] {query['id']:10s} NDCG@5={ndcg_5:.3f} "
                  f"MRR={mrr:.3f} Lat={latency_ms:.0f}ms — {query['query_text'][:50]}")

        # ----- Aggregates -----
        bench = RecommenderBenchmark(
            name=name,
            ingestion_latency_ms=ingestion_ms,
            query_results=query_results,
        )
        latencies_all = sorted(qr.latency_ms for qr in query_results)
        n = len(query_results)
        bench.mean_ndcg_5 = sum(qr.ndcg_at_5 for qr in query_results) / n
        bench.mean_ndcg_10 = sum(qr.ndcg_at_10 for qr in query_results) / n
        bench.mean_mrr = sum(qr.mrr for qr in query_results) / n
        bench.mean_attr_f1 = sum(qr.attribute_f1 for qr in query_results) / n
        bench.coverage = compute_coverage(query_results)
        bench.mean_explanation = sum(qr.explanation_score for qr in query_results) / n
        bench.latency_p50_ms = latencies_all[n // 2]
        bench.latency_p95_ms = latencies_all[int(n * 0.95)]
        bench.latency_p99_ms = latencies_all[int(n * 0.99)]

        results.append(bench)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recommendation System Benchmark")
    parser.add_argument(
        "--recommenders", type=Path, required=True,
        help="Path to designs/ directory containing recommender implementations",
    )
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parent / "data",
        help="Path to benchmark data directory",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Comma-separated list of design names to benchmark (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per measurement (default: 3, median used)",
    )
    args = parser.parse_args()

    print("Loading benchmark dataset...")
    dataset = load_dataset(args.data)
    print(f"  Ski products: {len(dataset['domains']['ski']['products'])}")
    print(f"  Ski reviews: {len(dataset['domains']['ski']['reviews'])}")
    print(f"  Shoe products: {len(dataset['domains']['running_shoe']['products'])}")
    print(f"  Shoe reviews: {len(dataset['domains']['running_shoe']['reviews'])}")
    print(f"  Queries: {len(dataset['queries'])}")

    filter_names = args.filter.split(",") if args.filter else None
    print(f"\nDiscovering recommenders in {args.recommenders}...")
    recommenders = discover_recommenders(args.recommenders, filter_names)

    if not recommenders:
        print("No recommenders found. Exiting.")
        return

    print(f"\nFound {len(recommenders)} recommender(s). Starting benchmark...")
    results = run_benchmark(recommenders, dataset, num_runs=args.runs)

    print("\nGenerating report...")
    args.output.mkdir(parents=True, exist_ok=True)
    generate_report(results, args.output)
    print(f"Results written to {args.output}/")


if __name__ == "__main__":
    main()
```

### Metrics Implementation

```python
"""
benchmark/metrics.py — Metric computation functions
"""

import math


def compute_ndcg(predicted_ids: list[str], gt_relevance: dict[str, int], k: int) -> float:
    """
    Compute NDCG@k.

    Args:
        predicted_ids: Ordered list of product IDs returned by the system
        gt_relevance: {product_id: relevance_grade} from ground truth (0-3 scale)
        k: Cutoff position

    Returns:
        NDCG@k score in [0, 1]
    """
    def dcg(relevances: list[float], k: int) -> float:
        return sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(relevances[:k])
        )

    # Actual DCG from predicted ranking
    actual_rels = [gt_relevance.get(pid, 0) for pid in predicted_ids]
    actual_dcg = dcg(actual_rels, k)

    # Ideal DCG from perfect ranking
    ideal_rels = sorted(gt_relevance.values(), reverse=True)
    ideal_dcg = dcg(ideal_rels, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_mrr(
    predicted_ids: list[str],
    gt_relevance: dict[str, int],
    threshold: int = 3,
) -> float:
    """
    Compute Reciprocal Rank for a single query.

    Returns 1/rank of the first result with relevance >= threshold.
    Returns 0 if no such result exists.
    """
    for i, pid in enumerate(predicted_ids):
        if gt_relevance.get(pid, 0) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def compute_attribute_precision(
    result_dicts: list[dict],
    expected_attributes: list[str],
) -> tuple[float, float, float]:
    """
    Compute attribute precision, recall, and F1.

    Compares the keys in each result's matched_attributes against the
    expected key_attributes for the query.

    Returns:
        (precision, recall, f1) averaged across all results
    """
    if not expected_attributes or not result_dicts:
        return (0.0, 0.0, 0.0)

    expected_set = set(attr.lower() for attr in expected_attributes)
    precisions = []
    recalls = []

    for result in result_dicts:
        matched = result.get("matched_attributes", {})
        if not matched:
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        predicted_set = set(k.lower() for k in matched.keys())
        intersection = predicted_set & expected_set

        p = len(intersection) / len(predicted_set) if predicted_set else 0.0
        r = len(intersection) / len(expected_set) if expected_set else 0.0
        precisions.append(p)
        recalls.append(r)

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    return (avg_p, avg_r, f1)


def compute_coverage(query_results) -> float:
    """
    Fraction of queries with at least one result with relevance >= 2.
    """
    covered = 0
    for qr in query_results:
        gt_map = {item["product_id"]: item["relevance"] for item in qr.ground_truth}
        for result in qr.results:
            if gt_map.get(result["product_id"], 0) >= 2:
                covered += 1
                break
    return covered / len(query_results) if query_results else 0.0


def compute_explanation_quality_proxy(
    result_dicts: list[dict],
    products: list[dict],
) -> float:
    """
    Automated proxy for explanation quality (1-5 scale).

    Checks whether explanations reference actual product attribute names.
    This is a heuristic — full evaluation requires human review.

    Scoring:
        5: mentions 3+ real attributes by name
        4: mentions 2 real attributes
        3: mentions 1 real attribute
        2: non-empty but no attribute match
        1: empty or missing
    """
    # Build attribute name lookup
    attribute_names = set()
    for product in products:
        for attr_name in product.get("attributes", {}).keys():
            attribute_names.add(attr_name.lower().replace("_", " "))
            attribute_names.add(attr_name.lower())

    # Common synonyms
    synonyms = {
        "stiffness": ["stiff", "stiffness", "flex"],
        "damp": ["damp", "dampness", "damping", "vibration"],
        "edge_grip": ["edge grip", "edge hold", "grip", "hold on ice"],
        "stability_at_speed": ["stability", "stable", "speed stability"],
        "playfulness": ["playful", "playfulness", "fun", "lively"],
        "powder_float": ["float", "powder", "flotation"],
        "forgiveness": ["forgiving", "forgiveness", "easy", "beginner-friendly"],
    }
    all_terms = set()
    for terms in synonyms.values():
        all_terms.update(terms)
    all_terms.update(attribute_names)

    scores = []
    for result in result_dicts:
        explanation = (result.get("explanation") or "").lower()
        if not explanation:
            scores.append(1.0)
            continue

        matches = sum(1 for term in all_terms if term in explanation)
        if matches >= 3:
            scores.append(5.0)
        elif matches >= 2:
            scores.append(4.0)
        elif matches >= 1:
            scores.append(3.0)
        else:
            scores.append(2.0)

    return sum(scores) / len(scores) if scores else 1.0
```

### Report Generation

```python
"""
benchmark/report.py — Report generation
"""

import json
from pathlib import Path
from dataclasses import asdict


def generate_report(results: list, output_dir: Path):
    """Generate summary table and detailed per-query results."""

    # ----- Summary table (text) -----
    lines = []
    lines.append("=" * 120)
    lines.append("RECOMMENDATION SYSTEM BENCHMARK — SUMMARY")
    lines.append("=" * 120)
    lines.append("")

    header = (
        f"{'Design':<20s} {'NDCG@5':>8s} {'NDCG@10':>8s} {'MRR':>8s} "
        f"{'AttrF1':>8s} {'Cover%':>8s} {'ExplQ':>8s} "
        f"{'Ingest':>10s} {'P50':>8s} {'P95':>8s} {'P99':>8s}"
    )
    lines.append(header)
    lines.append("-" * 120)

    # Sort by NDCG@5 descending
    sorted_results = sorted(results, key=lambda r: r.mean_ndcg_5, reverse=True)

    for i, bench in enumerate(sorted_results):
        rank_marker = " *" if i == 0 else ""
        line = (
            f"{bench.name:<20s} {bench.mean_ndcg_5:>8.3f} {bench.mean_ndcg_10:>8.3f} "
            f"{bench.mean_mrr:>8.3f} {bench.mean_attr_f1:>8.3f} "
            f"{bench.coverage * 100:>7.1f}% {bench.mean_explanation:>8.2f} "
            f"{bench.ingestion_latency_ms:>9.0f}ms {bench.latency_p50_ms:>7.0f}ms "
            f"{bench.latency_p95_ms:>7.0f}ms {bench.latency_p99_ms:>7.0f}ms{rank_marker}"
        )
        lines.append(line)

    lines.append("-" * 120)
    lines.append("")

    # ----- Per-difficulty breakdown -----
    difficulties = ["easy", "medium", "hard", "vague", "cross_domain"]
    lines.append("PER-DIFFICULTY NDCG@5 BREAKDOWN")
    lines.append("-" * 90)
    diff_header = f"{'Design':<20s}" + "".join(f"{d:>14s}" for d in difficulties)
    lines.append(diff_header)
    lines.append("-" * 90)

    for bench in sorted_results:
        parts = [f"{bench.name:<20s}"]
        for diff in difficulties:
            diff_queries = [qr for qr in bench.query_results if qr.difficulty == diff]
            if diff_queries:
                avg = sum(qr.ndcg_at_5 for qr in diff_queries) / len(diff_queries)
                parts.append(f"{avg:>14.3f}")
            else:
                parts.append(f"{'N/A':>14s}")
        lines.append("".join(parts))

    lines.append("")

    # ----- Worst queries per design -----
    lines.append("WEAKEST QUERIES PER DESIGN (lowest NDCG@5)")
    lines.append("-" * 90)
    for bench in sorted_results:
        worst = sorted(bench.query_results, key=lambda qr: qr.ndcg_at_5)[:3]
        lines.append(f"\n  {bench.name}:")
        for qr in worst:
            lines.append(
                f"    {qr.query_id:10s} NDCG@5={qr.ndcg_at_5:.3f} — {qr.query_text[:60]}"
            )

    lines.append("")
    lines.append("=" * 120)
    lines.append(f"Winner: {sorted_results[0].name} "
                 f"(NDCG@5 = {sorted_results[0].mean_ndcg_5:.3f})")
    lines.append("=" * 120)

    summary_text = "\n".join(lines)
    print(summary_text)

    # Write files
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary_text)

    # ----- JSON summary -----
    summary_json = []
    for bench in sorted_results:
        summary_json.append({
            "name": bench.name,
            "mean_ndcg_5": round(bench.mean_ndcg_5, 4),
            "mean_ndcg_10": round(bench.mean_ndcg_10, 4),
            "mean_mrr": round(bench.mean_mrr, 4),
            "mean_attr_f1": round(bench.mean_attr_f1, 4),
            "coverage": round(bench.coverage, 4),
            "mean_explanation_quality": round(bench.mean_explanation, 2),
            "ingestion_latency_ms": round(bench.ingestion_latency_ms, 1),
            "latency_p50_ms": round(bench.latency_p50_ms, 1),
            "latency_p95_ms": round(bench.latency_p95_ms, 1),
            "latency_p99_ms": round(bench.latency_p99_ms, 1),
            "per_query": [
                {
                    "query_id": qr.query_id,
                    "difficulty": qr.difficulty,
                    "ndcg_5": round(qr.ndcg_at_5, 4),
                    "ndcg_10": round(qr.ndcg_at_10, 4),
                    "mrr": round(qr.mrr, 4),
                    "attr_f1": round(qr.attribute_f1, 4),
                    "explanation_score": round(qr.explanation_score, 2),
                    "latency_ms": round(qr.latency_ms, 1),
                }
                for qr in bench.query_results
            ],
        })

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # ----- Per-query detail files -----
    detail_dir = output_dir / "per_query"
    detail_dir.mkdir(parents=True, exist_ok=True)

    for bench in results:
        for qr in bench.query_results:
            detail = {
                "recommender": bench.name,
                "query_id": qr.query_id,
                "query_text": qr.query_text,
                "difficulty": qr.difficulty,
                "domain": qr.domain,
                "metrics": {
                    "ndcg_5": round(qr.ndcg_at_5, 4),
                    "ndcg_10": round(qr.ndcg_at_10, 4),
                    "mrr": round(qr.mrr, 4),
                    "attribute_precision": round(qr.attribute_precision, 4),
                    "attribute_recall": round(qr.attribute_recall, 4),
                    "attribute_f1": round(qr.attribute_f1, 4),
                    "explanation_score": round(qr.explanation_score, 2),
                    "latency_ms": round(qr.latency_ms, 1),
                },
                "results": qr.results,
                "ground_truth": qr.ground_truth,
            }
            filename = f"{bench.name}_{qr.query_id}.json"
            with open(detail_dir / filename, "w") as f:
                json.dump(detail, f, indent=2)
```

---

## 6. How to Add a Recommender

Each recommender implementation must be a Python package under the `designs/` directory with a `create_recommender()` factory function:

```python
# designs/design_01/__init__.py

from dataclasses import dataclass


@dataclass
class RecommendationResult:
    product_id: str
    product_name: str
    score: float
    explanation: str
    matched_attributes: dict[str, float]


class MyRecommender:
    def ingest(self, products: list[dict], reviews: list[dict], domain: str) -> None:
        """Process and store product and review data for the given domain."""
        ...

    def query(self, query_text: str, domain: str, top_k: int = 10) -> list[RecommendationResult]:
        """Return top-k recommendations for the given query."""
        ...


def create_recommender():
    return MyRecommender()
```

---

## 7. Interpreting Results

### Composite Score

The final ranking of recommenders uses a weighted composite:

| Metric | Weight | Rationale |
|--------|--------|-----------|
| NDCG@5 | 0.30 | Primary ranking quality measure |
| NDCG@10 | 0.15 | Broader ranking quality |
| MRR | 0.15 | How quickly users find the best result |
| Attribute F1 | 0.10 | Correctness of attribute identification |
| Coverage | 0.10 | Robustness across query types |
| Explanation Quality | 0.10 | Usability of outputs |
| Latency (inverted) | 0.10 | Practical performance |

```python
def composite_score(bench):
    latency_score = max(0, 1 - bench.latency_p50_ms / 5000)  # 0-1, 5s = 0
    return (
        0.30 * bench.mean_ndcg_5
        + 0.15 * bench.mean_ndcg_10
        + 0.15 * bench.mean_mrr
        + 0.10 * bench.mean_attr_f1
        + 0.10 * bench.coverage
        + 0.10 * (bench.mean_explanation / 5.0)
        + 0.10 * latency_score
    )
```

### What "Winning" Looks Like

A strong recommender should:
- **NDCG@5 >= 0.7** on easy queries, **>= 0.5** on medium, **>= 0.3** on hard/vague
- **MRR >= 0.5** overall (first relevant result in top 2 on average)
- **Coverage = 100%** (every query returns at least one relevant result)
- **Query latency < 500ms** at p95
- Handle negations (HARD-01, HARD-02) and subjective language (VAGUE-01, VAGUE-02)
- Correctly switch domains for XDOM queries

### Red Flags

- NDCG@5 = 0 on any easy query indicates a fundamental issue
- Coverage below 80% suggests the system fails silently on some query types
- High variance between easy and hard queries may indicate keyword-only matching
- Explanation quality below 2.0 suggests hallucinated or generic explanations
- Latency above 2s per query may indicate architectural problems

---

## 8. Running the Benchmark

```bash
# Full benchmark
python benchmark/runner.py --recommenders designs/

# Specific designs only
python benchmark/runner.py --recommenders designs/ --filter "design_01,design_05"

# More measurement runs for stable latency numbers
python benchmark/runner.py --recommenders designs/ --runs 5

# Custom data/output directories
python benchmark/runner.py --recommenders designs/ --data benchmark/data/ --output benchmark/results/
```

Results are written to `benchmark/results/`:
- `summary.txt` — human-readable comparison table
- `summary.json` — machine-readable full results
- `per_query/{design}_{query_id}.json` — detailed per-query breakdown with returned results and ground truth
