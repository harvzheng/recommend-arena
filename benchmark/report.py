"""
benchmark/report.py — Report generation for the recommendation benchmark.
"""

import json
from pathlib import Path


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
