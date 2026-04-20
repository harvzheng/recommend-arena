"""
benchmark/runner.py — Main benchmark runner for recommendation systems.

Usage:
    python runner.py --recommenders path/to/designs/ --output results/
    python runner.py --recommenders path/to/designs/ --filter "design_01,design_05"
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass

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
    results: list[dict]
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

            # Convert results to dicts
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
