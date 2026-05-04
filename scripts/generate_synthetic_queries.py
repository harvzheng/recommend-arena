#!/usr/bin/env python3
"""generate_synthetic_queries.py — Step 1 of the domain compiler.

For each product, prompt a teacher LLM to generate ~10 queries spanning
the existing difficulty buckets (easy / medium / hard / vague). For each
generated query, identify 3-5 hard negatives by nearest-neighbor in
vanilla-embedding space among products that should *not* match.

Output: `<bundle>/synthetic_train.jsonl` with rows like
    {"query": "stiff freeride ski over 105mm waist that's not playful",
     "positive_id": "ski_023",
     "hard_negatives": ["ski_007", "ski_012", "ski_019"],
     "difficulty": "hard"}

Aim for ~10K triples per domain (the spec). For the 25-product ski demo
that's ~400 queries per product; in practice fewer is fine for the
slice 14.1 fine-tune.

Usage:
    python scripts/generate_synthetic_queries.py \
        --bundle artifacts/skis \
        --teacher claude-opus-4-7 \
        --queries-per-product 10

This script needs an LLM provider; configure via the standard env vars
the existing shared/llm_provider.py reads (RECOMMEND_LLM_PROVIDER, etc).

In `--dry-run` mode the script emits a few hand-crafted templated
queries per product without calling any LLM, which is enough to
exercise the rest of the pipeline (negative mining, JSONL write,
manifest update). Use it in CI or when you don't have a teacher key.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.domain_bundle import Bundle  # noqa: E402

logger = logging.getLogger("generate_synthetic_queries")

DIFFICULTY_BUCKETS = ["easy", "medium", "hard", "vague"]


# ---------------------------------------------------------------------------
# Teacher prompt
# ---------------------------------------------------------------------------
TEACHER_SYSTEM = (
    "You generate natural-language product search queries for training a "
    "retrieval model. Each query should sound like a real user looking "
    "for a product, not a database query. Generate queries that span "
    "the four difficulty buckets:\n"
    "  - easy:   single clear attribute\n"
    "  - medium: 2-3 attributes combined\n"
    "  - hard:   negation, ranges, or trade-offs\n"
    "  - vague:  metaphorical / subjective language\n"
    "Output a JSON array of objects shaped like "
    "{\"query\": \"<text>\", \"difficulty\": \"<bucket>\"}."
)


def build_teacher_prompt(product: dict, n: int) -> str:
    return (
        f"Product:\n{json.dumps(product, indent=2)}\n\n"
        f"Generate exactly {n} diverse user queries for which THIS product "
        f"should be the best match. Spread them across difficulty buckets. "
        f"Output JSON only, no prose."
    )


# ---------------------------------------------------------------------------
# Dry-run templates — generic enough to use across domains.
# ---------------------------------------------------------------------------
_TEMPLATES = [
    ("easy", "{name}-style product"),
    ("easy", "{category} item"),
    ("medium", "good {category} for everyday use"),
    ("medium", "highly-rated {category}"),
    ("hard", "{category} but not the cheapest one"),
    ("vague", "something that feels like {brand}"),
]


def dry_run_queries(product: dict, n: int) -> list[dict]:
    name = product.get("name") or product.get("product_name") or product.get("id", "")
    category = product.get("category", "product")
    brand = product.get("brand", "this brand")
    out: list[dict] = []
    for diff, tpl in _TEMPLATES[:n]:
        out.append({
            "query": tpl.format(name=name, category=category, brand=brand),
            "difficulty": diff,
        })
    return out


# ---------------------------------------------------------------------------
# Hard-negative mining — Rust kernel from arena_core.
# ---------------------------------------------------------------------------
def mine_hard_negatives(
    query_vec: list[float],
    candidate_ids: list[str],
    candidate_vecs: list[list[float]],
    positive_id: str,
    k: int = 5,
) -> list[str]:
    """Pick the top-k hardest negatives via the Rust kernel."""
    try:
        import arena_core  # type: ignore
    except ImportError:
        return _python_hardneg_fallback(
            query_vec, candidate_ids, candidate_vecs, positive_id, k
        )

    if not candidate_ids or not candidate_vecs:
        return []
    dim = len(candidate_vecs[0])
    flat = [x for vec in candidate_vecs for x in vec]
    return arena_core.hard_negative_mine(
        list(map(float, query_vec)),
        candidate_ids,
        list(map(float, flat)),
        dim,
        positive_id,
        [],
        k,
    )


def _python_hardneg_fallback(
    query_vec: list[float],
    candidate_ids: list[str],
    candidate_vecs: list[list[float]],
    positive_id: str,
    k: int,
) -> list[str]:
    scored: list[tuple[str, float]] = []
    for pid, cv in zip(candidate_ids, candidate_vecs):
        if pid == positive_id:
            continue
        s = sum(a * b for a, b in zip(query_vec, cv))
        scored.append((pid, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:k]]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def encode_products(products: list[dict], bundle_domain: str) -> dict[str, list[float]]:
    """Encode each product so we can mine hard negatives. Lazy-imports
    sentence-transformers; in dry-run we substitute zero-vectors."""
    try:
        from implementations.design_14_local_hybrid import retrieval
    except ImportError:
        return {}

    return retrieval.encode_products(products, [], retrieval.DEFAULT_EMBED_MODEL)


def encode_queries(queries: list[str]) -> list[list[float]]:
    try:
        from implementations.design_14_local_hybrid import retrieval
    except ImportError:
        return [[0.0] * 1024 for _ in queries]
    return [retrieval.encode_query(q) or [0.0] * 1024 for q in queries]


def call_teacher(prompt: str) -> list[dict]:
    """Call the teacher LLM via shared.llm_provider, parse JSON output."""
    from shared.llm_provider import get_provider
    provider = get_provider()
    text = provider.generate(
        f"{TEACHER_SYSTEM}\n\n{prompt}",
        json_mode=True,
    )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("teacher returned non-JSON; skipping product: %s", e)
        return []
    if not isinstance(data, list):
        logger.warning("teacher output is not a JSON array; skipping product")
        return []
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = item.get("query")
        d = item.get("difficulty", "medium")
        if isinstance(q, str) and q.strip():
            out.append({"query": q.strip(), "difficulty": d})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="path to bundle dir")
    parser.add_argument("--queries-per-product", type=int, default=10)
    parser.add_argument(
        "--hard-negatives-per-query", type=int, default=5,
        help="k for hard-negative mining",
    )
    parser.add_argument(
        "--teacher", default=os.environ.get("ARENA_TEACHER", "claude-opus-4-7"),
        help="teacher model name (passed to LLMProvider via env)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="skip the LLM call; emit templated queries (CI / no-API-key mode)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="seed for the dry-run sampler",
    )
    parser.add_argument(
        "--max-products", type=int, default=None,
        help="cap how many products we generate queries for (debug)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )
    random.seed(args.seed)

    bundle = Bundle.load(args.bundle)
    products = bundle.read_products()
    if args.max_products:
        products = products[: args.max_products]
    logger.info("generating queries for %d products", len(products))

    # Encode the catalog once. In dry-run mode this still works (zero vectors)
    # but the hard negatives are deterministic by catalog index, which is fine
    # for an end-to-end smoke test.
    if args.dry_run:
        logger.info("dry-run: skipping encoder; using zero vectors")
        product_vecs = {
            (p.get("id") or p.get("product_id") or ""): [0.0] * 1024
            for p in products
        }
    else:
        logger.info("encoding catalog with default embedding model …")
        product_vecs = encode_products(products, bundle.manifest.domain)
        if not product_vecs:
            logger.error(
                "no encoder available; install sentence-transformers, "
                "or pass --dry-run for templated queries"
            )
            return 2

    candidate_ids = [
        p.get("id") or p.get("product_id") or "" for p in products
    ]
    candidate_vecs_list = [product_vecs.get(pid, [0.0] * 1024) for pid in candidate_ids]

    # Generate per-product queries.
    triples: list[dict] = []
    for product in products:
        pid = product.get("id") or product.get("product_id") or ""
        if not pid:
            continue

        if args.dry_run:
            queries = dry_run_queries(product, args.queries_per_product)
        else:
            prompt = build_teacher_prompt(product, args.queries_per_product)
            queries = call_teacher(prompt)
            if not queries:
                logger.warning("no queries for product %s", pid)
                continue

        # Encode the queries in a single batch when possible.
        if args.dry_run:
            qvecs = [[0.0] * 1024 for _ in queries]
        else:
            qvecs = encode_queries([q["query"] for q in queries])

        for q_obj, qvec in zip(queries, qvecs):
            negs = mine_hard_negatives(
                qvec,
                candidate_ids,
                candidate_vecs_list,
                pid,
                k=args.hard_negatives_per_query,
            )
            triples.append({
                "query": q_obj["query"],
                "positive_id": pid,
                "hard_negatives": negs,
                "difficulty": q_obj.get("difficulty", "medium"),
            })

    out_path = bundle.paths.root / "synthetic_train.jsonl"
    with out_path.open("w") as f:
        for t in triples:
            f.write(json.dumps(t, sort_keys=True))
            f.write("\n")
    logger.info("wrote %d triples to %s", len(triples), out_path)

    bundle.manifest.metadata.setdefault("synthetic", {})
    bundle.manifest.metadata["synthetic"] = {
        "n_triples": len(triples),
        "queries_per_product": args.queries_per_product,
        "hard_negatives_per_query": args.hard_negatives_per_query,
        "teacher": args.teacher,
        "dry_run": args.dry_run,
    }
    bundle.save_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
