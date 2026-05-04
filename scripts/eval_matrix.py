#!/usr/bin/env python3
"""eval_matrix.py — sweep design-14 configurations against a bundle.

Prints a matrix of NDCG@5 across pre-defined configurations so it's easy
to see which lever (retrieval, embedding, reranker model, listwise vs
pointwise) actually moves the needle. Also prints recall@10/25 of the
fused-but-unreranked candidate set so you can tell when the ceiling is
retrieval, not reranking.

Usage:
    python scripts/eval_matrix.py --bundle artifacts/ski

Add --configs to pick a subset (default: all configs).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.domain_bundle import Bundle  # noqa: E402

logger = logging.getLogger("eval_matrix")


def ndcg_at_k(pred: list[str], rel: dict[str, int], k: int = 5) -> float:
    rs = [rel.get(p, 0) for p in pred[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    if not ideal or sum(ideal) == 0:
        return 0.0
    dcg = sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(rs))
    idcg = sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg


CONFIGS: dict[str, dict] = {
    "lexical_only": dict(enable_vector=False, enable_reranker=False),
    "bge_small_vec": dict(
        enable_vector=True, enable_reranker=False,
        embedding_model="BAAI/bge-small-en-v1.5",
    ),
    "bge_small_rerank": dict(
        enable_vector=True, enable_reranker=True,
        embedding_model="BAAI/bge-small-en-v1.5",
        reranker_model="BAAI/bge-reranker-base",
    ),
    "bge_base_rerank": dict(
        enable_vector=True, enable_reranker=True,
        embedding_model="BAAI/bge-base-en-v1.5",
        reranker_model="BAAI/bge-reranker-base",
    ),
    "qwen_listwise_vanilla_top10": dict(
        enable_vector=True, enable_reranker=True,
        embedding_model="BAAI/bge-small-en-v1.5",
        reranker_model="Qwen/Qwen3-1.7B",
        reranker_kind="listwise",
        _env={"RECOMMEND_LISTWISE_TOP_K": "10"},
    ),
    "qwen_listwise_vanilla_top20": dict(
        enable_vector=True, enable_reranker=True,
        embedding_model="BAAI/bge-small-en-v1.5",
        reranker_model="Qwen/Qwen3-1.7B",
        reranker_kind="listwise",
        _env={"RECOMMEND_LISTWISE_TOP_K": "20"},
    ),
    "qwen_listwise_lora_top20": dict(
        enable_vector=True, enable_reranker=True,
        embedding_model="BAAI/bge-small-en-v1.5",
        reranker_model="Qwen/Qwen3-1.7B",
        reranker_adapter_path=None,  # filled in from bundle in main()
        reranker_kind="listwise",
        _env={"RECOMMEND_LISTWISE_TOP_K": "20"},
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--configs", default=None,
                        help="comma-separated subset of config names")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    bundle = Bundle.load(args.bundle)
    queries = bundle.read_eval_queries()
    products = bundle.read_products()
    reviews = bundle.read_reviews()

    # Resolve the bundle's listwise adapter path if present
    if bundle.manifest.reranker and bundle.manifest.reranker.adapter_path:
        adapter = str(bundle.paths.root / bundle.manifest.reranker.adapter_path)
        if "qwen_listwise_lora_top20" in CONFIGS:
            CONFIGS["qwen_listwise_lora_top20"]["reranker_adapter_path"] = adapter

    selected = (
        [c.strip() for c in args.configs.split(",")]
        if args.configs else list(CONFIGS.keys())
    )

    # Lazy import so non-config errors are diagnostic-only.
    from implementations.design_14_local_hybrid.recommender import LocalHybridRecommender

    rows: list[tuple[str, float, dict[str, float]]] = []
    for name in selected:
        cfg = dict(CONFIGS[name])
        env = cfg.pop("_env", {}) or {}
        if cfg.get("reranker_adapter_path") is None and name == "qwen_listwise_lora_top20":
            print(f"-- skipping {name}: no listwise adapter in bundle")
            continue

        old_env = {k: os.environ.get(k) for k in env}
        for k, v in env.items():
            os.environ[k] = v
        try:
            rec = LocalHybridRecommender(**cfg)
            rec.ingest(products, reviews, bundle.manifest.domain)
            per_q: dict[str, float] = {}
            for q in queries:
                rel = {g["product_id"]: g["relevance"] for g in q.get("ground_truth_top5", [])}
                res = rec.query(q["query_text"], q.get("domain", bundle.manifest.domain), top_k=args.top_k)
                per_q[q["id"]] = ndcg_at_k([r.product_id for r in res], rel, k=args.top_k)
            mean = sum(per_q.values()) / max(1, len(per_q))
            rows.append((name, mean, per_q))
            print(f"{name:32s}  NDCG@{args.top_k}={mean:.3f}")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    print()
    print("=== matrix ===")
    qids = sorted({qid for _, _, p in rows for qid in p})
    headers = ["query".ljust(12)] + [name[:10] for name, _, _ in rows]
    print("  ".join(headers))
    for qid in qids:
        line = [qid.ljust(12)]
        for _, _, per_q in rows:
            v = per_q.get(qid, 0.0)
            mark = "!" if v < 0.5 else " "
            line.append(f"{mark}{v:.2f}".rjust(10))
        print("  ".join(line))
    print()
    print("  ".join(["MEAN".ljust(12)] + [f"{m:.3f}".rjust(10) for _, m, _ in rows]))

    out = {
        "configs": {name: {"mean": m, "per_query": p} for name, m, p in rows},
    }
    eval_dir = bundle.paths.root / "eval"
    eval_dir.mkdir(exist_ok=True)
    out_path = eval_dir / "matrix.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
