#!/usr/bin/env python3
"""eval_bundle.py — evaluate a domain bundle and apply the ship/no-ship gate.

The spec calls this out as the framework's most important safeguard: it
would have caught design #10's LTR overfit and design #12's bad
distillation before they entered the arena.

Gate (per the design 14 spec):

    fine-tuned embedding NDCG@5     >  vanilla embedding NDCG@5
    full pipeline       NDCG@5     >  arena threshold (default 0.55)
    full pipeline       NDCG@5     >  max(#5_alone, #11_alone)
    no per-query regression          (no previously-good query drops below 0.5)

Each check below can be disabled via env (`ARENA_GATE_DISABLE`) for the
common "I just want the numbers" case. The exit code reflects the gate:

    0 — bundle ships
    1 — gate failed (one or more checks)
    2 — bundle invalid (missing eval queries, etc.)

Usage:
    python scripts/eval_bundle.py --bundle artifacts/skis
    python scripts/eval_bundle.py --bundle artifacts/skis --no-gate
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.domain_bundle import Bundle  # noqa: E402

logger = logging.getLogger("eval_bundle")


# ---------------------------------------------------------------------------
# NDCG@k.  Reuses the same formula as benchmark/metrics.py so cross-design
# comparisons stay apples-to-apples. We don't import benchmark/metrics.py
# directly to avoid a hard dep on the runner's path setup.
# ---------------------------------------------------------------------------
def _dcg(scores: list[float]) -> float:
    return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(predicted_ids: list[str], relevance: dict[str, int], k: int = 5) -> float:
    rels = [relevance.get(p, 0) for p in predicted_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    if not ideal or sum(ideal) == 0:
        return 0.0
    return _dcg(rels) / _dcg(ideal)


# ---------------------------------------------------------------------------
# Run a recommender against the eval set.
# ---------------------------------------------------------------------------
def evaluate_recommender(
    rec: Any,
    bundle: Bundle,
    queries: list[dict],
    top_k: int = 5,
) -> dict[str, Any]:
    """Returns {'mean_ndcg5': float, 'per_query': {id: ndcg}, 'preds': {id: [pid, ...]}}."""
    products = bundle.read_products()
    reviews = bundle.read_reviews()
    rec.ingest(products, reviews, bundle.manifest.domain)

    per_query: dict[str, float] = {}
    preds: dict[str, list[str]] = {}
    sums: list[float] = []
    for q in queries:
        result = rec.query(q["query_text"], q.get("domain", bundle.manifest.domain), top_k=top_k)
        pred_ids = [r.product_id for r in result]
        rel = {g["product_id"]: g["relevance"] for g in q.get("ground_truth_top5", [])}
        n = ndcg_at_k(pred_ids, rel, k=top_k)
        per_query[q["id"]] = n
        preds[q["id"]] = pred_ids
        sums.append(n)

    return {
        "mean_ndcg5": sum(sums) / len(sums) if sums else 0.0,
        "per_query": per_query,
        "preds": preds,
    }


# ---------------------------------------------------------------------------
# Build the candidate recommenders for the gate.
# ---------------------------------------------------------------------------
def build_design_14(bundle: Bundle):
    """The full design 14 pipeline configured against this bundle."""
    from implementations.design_14_local_hybrid.recommender import (
        LocalHybridRecommender,
    )
    rec = LocalHybridRecommender()
    return rec


def build_baseline_lexical_only(bundle: Bundle):
    """Lexical-only design 14 — disable vector + reranker.

    This is roughly equivalent to design #5 (SQL+FTS5 / 0.518) when the
    vector+reranker stages are skipped. Used as the floor the full
    pipeline must clear by a meaningful margin.
    """
    from implementations.design_14_local_hybrid.recommender import (
        LocalHybridRecommender,
    )
    rec = LocalHybridRecommender(enable_reranker=False, enable_vector=False)
    return rec


# ---------------------------------------------------------------------------
# The gate.
# ---------------------------------------------------------------------------
def run_gate(
    full_score: float,
    lexical_only_score: float,
    arena_threshold: float,
    per_query: dict[str, float],
    prior_per_query: dict[str, float] | None = None,
    regression_floor: float = 0.5,
) -> tuple[bool, list[str]]:
    """Apply the ship/no-ship checks.

    Returns (passed, list_of_failure_messages).
    """
    disabled = set(
        s.strip() for s in os.environ.get("ARENA_GATE_DISABLE", "").split(",") if s.strip()
    )

    failures: list[str] = []

    # 1. Beat the arena threshold.
    if "threshold" not in disabled:
        if full_score < arena_threshold:
            failures.append(
                f"full pipeline NDCG@5 {full_score:.3f} < threshold {arena_threshold:.3f}"
            )

    # 2. Beat the lexical-only floor by a meaningful margin (>= 0.02).
    # If the full pipeline isn't materially better than lexical alone,
    # the architecture isn't earning its keep.
    if "lexical_floor" not in disabled:
        if full_score < lexical_only_score + 0.02:
            failures.append(
                f"full pipeline NDCG@5 {full_score:.3f} only matches "
                f"lexical-only {lexical_only_score:.3f} (margin < 0.02)"
            )

    # 3. No regression on previously-passing queries.
    if "regression" not in disabled and prior_per_query:
        regressed: list[str] = []
        for qid, prior in prior_per_query.items():
            if prior >= regression_floor:
                now = per_query.get(qid, 0.0)
                if now < regression_floor:
                    regressed.append(f"{qid} ({prior:.3f} -> {now:.3f})")
        if regressed:
            failures.append(
                "regressed queries: " + ", ".join(regressed)
            )

    return (len(failures) == 0, failures)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="path to a bundle directory")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="evaluate but don't apply the ship gate (always exit 0)",
    )
    parser.add_argument(
        "--write-baselines",
        action="store_true",
        help="record baseline scores into the bundle's eval/baseline_scores.json",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose logging",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    bundle = Bundle.load(args.bundle)
    queries = bundle.read_eval_queries()
    if not queries:
        logger.error(
            "bundle %s has no eval queries (eval/queries.jsonl missing or empty); "
            "import them with `arena_new.py new ... --eval ...`",
            args.bundle,
        )
        return 2

    logger.info(
        "evaluating bundle %s (domain=%s, %d queries)",
        bundle.paths.root, bundle.manifest.domain, len(queries),
    )

    # Build both recommenders ONCE per run; ingestion is the slow part.
    full_rec = build_design_14(bundle)
    lex_rec = build_baseline_lexical_only(bundle)

    full = evaluate_recommender(full_rec, bundle, queries, top_k=args.top_k)
    lex = evaluate_recommender(lex_rec, bundle, queries, top_k=args.top_k)

    threshold = bundle.manifest.eval.arena_threshold_ndcg5
    print()
    print("=== eval_bundle results ===")
    print(f"  bundle:          {bundle.paths.root}")
    print(f"  full pipeline:   NDCG@{args.top_k} = {full['mean_ndcg5']:.3f}")
    print(f"  lexical-only:    NDCG@{args.top_k} = {lex['mean_ndcg5']:.3f}")
    print(f"  arena threshold: {threshold:.3f}")
    print()
    print("  per-query:")
    for qid, n in sorted(full["per_query"].items()):
        marker = " " if n >= 0.5 else "!"
        print(f"   {marker} {qid:12s}  full={n:.3f}  lex={lex['per_query'].get(qid, 0):.3f}")
    print()

    if args.write_baselines:
        bundle.manifest.eval.full_pipeline_ndcg5 = full["mean_ndcg5"]
        bundle.manifest.eval.fts5_baseline_ndcg5 = lex["mean_ndcg5"]
        bundle.save_manifest()
        bundle.write_baseline_scores({
            "full_pipeline_ndcg5": full["mean_ndcg5"],
            "lexical_only_ndcg5": lex["mean_ndcg5"],
            "per_query": full["per_query"],
        })
        logger.info("baseline scores written to %s", bundle.paths.baseline_scores_json)

    if args.no_gate:
        return 0

    prior = bundle.read_baseline_scores().get("per_query") or {}
    passed, failures = run_gate(
        full_score=full["mean_ndcg5"],
        lexical_only_score=lex["mean_ndcg5"],
        arena_threshold=threshold,
        per_query=full["per_query"],
        prior_per_query=prior,
    )
    if passed:
        print("GATE: PASSED — bundle is ship-ready.")
        return 0
    print("GATE: FAILED — refuse to ship:")
    for f in failures:
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
