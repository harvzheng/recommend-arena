#!/usr/bin/env python3
"""arena new <domain> — produce a domain bundle from raw catalog/reviews.

Usage:
    python scripts/arena_new.py new <domain> \
        --catalog path/to/products.jsonl \
        --reviews path/to/reviews.jsonl \
        [--eval path/to/eval.jsonl] \
        [--out artifacts/<domain>] \
        [--embedding off_the_shelf|skip] \
        [--reranker off_the_shelf|skip]

Slice 14.0 (this commit) covers the *skeleton*: directory layout,
manifest, ingestion of an existing domain, and a vanilla embedding /
reranker pointer. Fine-tuning steps land in slices 14.1 and 14.2 and
plug into this same CLI as additional `--steps` flags.

Pipeline (per the design 14 spec; ✓ = wired in 14.0):

    [✓] step 0  ingest_dataset:        copy catalog + reviews into bundle
    [✓] step 0a build_fts5:            compile the SQLite + FTS5 index
    [✓] step 0b vanilla_embedding_pin: record off-the-shelf embedding in manifest
    [✓] step 0c vanilla_reranker_pin:  record off-the-shelf reranker in manifest
    [ ] step 1  generate_synthetic:    teacher LLM → synthetic_train.jsonl
    [ ] step 2  finetune_embedding:    train Qwen3-Embedding-0.6B on the triples
    [ ] step 4  finetune_reranker:     train bge-reranker LoRA on the triples
    [ ] step 5  extract_schema:        Pydantic schema.py
    [✓] step 6  eval_gate:             baselines + threshold check (in eval_bundle.py)

The --steps flag lets you run a subset; default is "ingest" only, which
is the action #3 deliverable. To run training, install the matching
requirements and pass --steps ingest,finetune-embedding (slice 14.1).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# Sibling scripts (extract_schema, generate_synthetic_queries, etc.) are
# imported by the step dispatchers below.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from shared.domain_bundle import (  # noqa: E402
    Bundle,
    EmbeddingArtifact,
    RerankerArtifact,
)

logger = logging.getLogger("arena_new")


# Off-the-shelf model IDs the spec calls out. Override at the CLI when
# experimenting with alternates (e.g. BGE-small-en-v1.5 instead).
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def cmd_new(args: argparse.Namespace) -> int:
    """The `arena new <domain>` entrypoint."""
    out = Path(args.out or f"artifacts/{args.domain}").resolve()
    catalog_path = Path(args.catalog).resolve()
    reviews_path = Path(args.reviews).resolve()

    products = _read_jsonl_or_json(catalog_path, key="products")
    reviews = _read_jsonl_or_json(reviews_path, key="reviews")
    logger.info(
        "loaded %d products + %d reviews for domain %r",
        len(products), len(reviews), args.domain,
    )

    if out.exists() and any(out.iterdir()):
        if not args.force:
            logger.error(
                "%s already exists and is non-empty; pass --force to overwrite",
                out,
            )
            return 2

    bundle = Bundle.create(out, args.domain)
    logger.info("created bundle at %s", bundle.paths.root)

    steps = set(args.steps.split(","))
    if "ingest" in steps:
        _step_ingest(bundle, products, reviews)
    if "embedding-pin" in steps and args.embedding != "skip":
        _step_pin_embedding(bundle, args.embedding_model)
    if "reranker-pin" in steps and args.reranker != "skip":
        _step_pin_reranker(bundle, args.reranker_model)
    if "eval-import" in steps and args.eval:
        _step_import_eval(bundle, Path(args.eval).resolve())

    # Slices 14.1 / 14.2 — dispatch to the dedicated step scripts.
    # Each is a clean subprocess so the heavy ML deps load only when
    # actually used.
    if "extract-schema" in steps:
        _step_extract_schema(bundle, args.schema_strategy)
    if "generate-synthetic" in steps:
        _step_generate_synthetic(
            bundle,
            queries_per_product=args.queries_per_product,
            hard_neg_k=args.hard_negatives_per_query,
            dry_run=args.synthetic_dry_run,
        )
    if "finetune-embedding" in steps:
        _step_finetune_embedding(bundle, dry_run=args.training_dry_run)
    if "finetune-reranker" in steps:
        _step_finetune_reranker(bundle, dry_run=args.training_dry_run)
    if "finetune-listwise" in steps:
        _step_finetune_listwise(
            bundle,
            base_model=args.listwise_base_model,
            dry_run=args.training_dry_run,
        )
    if "discover-filters" in steps:
        _step_discover_filters(bundle)

    # The slice 14.1 / 14.2 dispatchers run as subprocess-style mains
    # that load their own Bundle handle and save it. Reload from disk
    # before the final save so we don't overwrite their changes with
    # the parent's stale in-memory copy.
    if any(s in steps for s in ("extract-schema", "generate-synthetic",
                                 "finetune-embedding", "finetune-reranker",
                                 "finetune-listwise", "discover-filters")):
        bundle = Bundle.load(bundle.paths.root)
    bundle.save_manifest()
    logger.info("manifest written to %s", bundle.paths.manifest_path)
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    """`arena pack <bundle> --out foo.tar.gz` — wire-format export."""
    bundle = Bundle.load(args.path)
    out = Path(args.out or f"{bundle.manifest.domain}.tar.gz").resolve()
    archive = bundle.pack(out)
    logger.info("packed bundle to %s (%d bytes)", archive, archive.stat().st_size)
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    """`arena unpack foo.tar.gz --into artifacts/` — wire-format import."""
    bundle = Bundle.unpack(args.archive, args.into)
    logger.info("unpacked bundle to %s", bundle.paths.root)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """`arena inspect <bundle>` — pretty-print the manifest + key file sizes."""
    bundle = Bundle.load(args.path)
    print(f"bundle:  {bundle.paths.root}")
    print(f"domain:  {bundle.manifest.domain}")
    print(f"format:  manifest_version {bundle.manifest.manifest_version}")
    print(f"counts:  {bundle.manifest.n_products} products, "
          f"{bundle.manifest.n_reviews} reviews")
    if bundle.manifest.embedding:
        e = bundle.manifest.embedding
        print(f"embed:   {e.kind} ({e.base_model}, dim={e.dim})")
    else:
        print("embed:   <not pinned>")
    if bundle.manifest.reranker:
        r = bundle.manifest.reranker
        print(f"rerank:  {r.kind} ({r.base_model})")
    else:
        print("rerank:  <not pinned>")
    ev = bundle.manifest.eval
    print(f"eval:    fts5={ev.fts5_baseline_ndcg5}  "
          f"vanilla_embed={ev.vanilla_embedding_baseline_ndcg5}  "
          f"full={ev.full_pipeline_ndcg5}  "
          f"threshold={ev.arena_threshold_ndcg5}")
    print()
    print("on-disk files:")
    for p in sorted(Path(bundle.paths.root).rglob("*")):
        if p.is_file():
            rel = p.relative_to(bundle.paths.root)
            print(f"  {rel}  ({p.stat().st_size:,} bytes)")
    return 0


# ---------------------------------------------------------------------------
# Compiler steps
# ---------------------------------------------------------------------------
def _step_ingest(bundle: Bundle, products: list[dict], reviews: list[dict]) -> None:
    """Step 0: copy raw data into the bundle and build the FTS5 index."""
    bundle.write_products(products)
    bundle.write_reviews(reviews)

    # Build the per-domain SQLite + FTS5 index in the same shape design 14
    # expects. We import retrieval lazily so this script can run before
    # the user has installed sentence-transformers.
    from implementations.design_14_local_hybrid import retrieval

    db_path = bundle.paths.fts5_db
    conn = retrieval.open_db(str(db_path))
    try:
        retrieval.ingest_products(conn, products, reviews)
    finally:
        conn.close()
    logger.info("built FTS5 index at %s", db_path)


def _step_pin_embedding(bundle: Bundle, model_id: str) -> None:
    """Step 0b: record an off-the-shelf embedding in the manifest.

    Doesn't download or cache the model — that happens lazily at runtime
    via sentence-transformers. We only record WHICH model the bundle
    uses so the eval gate and later fine-tunes are reproducible.
    """
    bundle.manifest.embedding = EmbeddingArtifact(
        kind="off_the_shelf",
        base_model=model_id,
        adapter_path=None,
        dim=_known_dim(model_id),
    )
    # Drop a small marker file so `arena inspect` shows something even
    # before the model is downloaded.
    bundle.paths.embedding_dir.mkdir(parents=True, exist_ok=True)
    (bundle.paths.embedding_dir / "model_id.txt").write_text(model_id + "\n")
    logger.info("pinned embedding: %s", model_id)


def _step_pin_reranker(bundle: Bundle, model_id: str) -> None:
    """Step 0c: record an off-the-shelf reranker in the manifest."""
    bundle.manifest.reranker = RerankerArtifact(
        kind="off_the_shelf",
        base_model=model_id,
        adapter_path=None,
    )
    bundle.paths.reranker_dir.mkdir(parents=True, exist_ok=True)
    (bundle.paths.reranker_dir / "model_id.txt").write_text(model_id + "\n")
    logger.info("pinned reranker: %s", model_id)


def _step_import_eval(bundle: Bundle, src: Path) -> None:
    """Copy a hand-curated eval set into the bundle."""
    queries = _read_jsonl_or_json(src, key="queries")
    bundle.write_eval_queries(queries)
    logger.info("imported %d eval queries from %s", len(queries), src)


# ---------------------------------------------------------------------------
# Slice 14.1 / 14.2 step dispatchers
# ---------------------------------------------------------------------------
def _step_extract_schema(bundle: Bundle, strategy: str) -> None:
    """Step 5: emit a Pydantic filter schema."""
    from extract_schema import main as extract_main
    rc = extract_main([
        "--bundle", str(bundle.paths.root),
        "--strategy", strategy,
    ])
    if rc != 0:
        logger.warning("extract_schema returned rc=%d", rc)


def _step_generate_synthetic(
    bundle: Bundle,
    queries_per_product: int,
    hard_neg_k: int,
    dry_run: bool,
) -> None:
    """Step 1: synthetic query generation + hard-negative mining."""
    from generate_synthetic_queries import main as gen_main
    argv = [
        "--bundle", str(bundle.paths.root),
        "--queries-per-product", str(queries_per_product),
        "--hard-negatives-per-query", str(hard_neg_k),
    ]
    if dry_run:
        argv.append("--dry-run")
    rc = gen_main(argv)
    if rc != 0:
        logger.warning("generate_synthetic_queries returned rc=%d", rc)


def _step_finetune_embedding(bundle: Bundle, dry_run: bool) -> None:
    """Step 2: embedding fine-tune."""
    from finetune_embedding import main as ft_main
    argv = ["--bundle", str(bundle.paths.root)]
    if dry_run:
        argv.append("--dry-run")
    rc = ft_main(argv)
    if rc != 0:
        logger.warning("finetune_embedding returned rc=%d", rc)


def _step_finetune_reranker(bundle: Bundle, dry_run: bool) -> None:
    """Step 4: reranker LoRA fine-tune."""
    from finetune_reranker import main as ft_main
    argv = ["--bundle", str(bundle.paths.root)]
    if dry_run:
        argv.append("--dry-run")
    rc = ft_main(argv)
    if rc != 0:
        logger.warning("finetune_reranker returned rc=%d", rc)


def _step_finetune_listwise(
    bundle: Bundle, base_model: str | None, dry_run: bool,
) -> None:
    """Step 4b: listwise reranker LoRA fine-tune."""
    from finetune_listwise import main as ft_main
    argv = ["--bundle", str(bundle.paths.root)]
    if base_model:
        argv.extend(["--base-model", base_model])
    if dry_run:
        argv.append("--dry-run")
    rc = ft_main(argv)
    if rc != 0:
        logger.warning("finetune_listwise returned rc=%d", rc)


def _step_discover_filters(bundle: Bundle) -> None:
    """Layer 2 of filter automation: LLM-discovered phrase mappings."""
    from discover_filter_phrases import main as df_main
    rc = df_main(["--bundle", str(bundle.paths.root)])
    if rc != 0:
        logger.warning("discover_filter_phrases returned rc=%d", rc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_jsonl_or_json(path: Path, key: str | None = None) -> list[dict]:
    """Read either a JSONL file or a JSON file with the data under `key`."""
    if path.suffix == ".jsonl":
        out: list[dict] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and key and key in data:
        return data[key]
    raise ValueError(
        f"{path}: expected JSONL or a JSON object with key {key!r}; "
        f"got {type(data).__name__}"
    )


# Known dims for the models the spec calls out. Used so the manifest
# carries the dim even before the model is downloaded.
_KNOWN_DIMS = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


def _known_dim(model_id: str) -> int:
    return _KNOWN_DIMS.get(model_id, 1024)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena", description=__doc__)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="verbose logging",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # `arena new`
    new = sub.add_parser("new", help="produce a domain bundle from raw data")
    new.add_argument("domain", help="domain name (used as artifacts/<domain>)")
    new.add_argument("--catalog", required=True, help="JSONL or JSON of products")
    new.add_argument("--reviews", required=True, help="JSONL or JSON of reviews")
    new.add_argument("--eval", default=None, help="optional hand-graded eval set")
    new.add_argument("--out", default=None, help="output dir (default: artifacts/<domain>)")
    new.add_argument("--force", action="store_true", help="overwrite existing bundle")
    new.add_argument(
        "--steps",
        default="ingest,embedding-pin,reranker-pin,eval-import",
        help="comma-separated subset of steps to run",
    )
    new.add_argument(
        "--embedding", default="off_the_shelf",
        choices=["off_the_shelf", "skip"],
        help="embedding strategy (slice 14.0 supports off_the_shelf only)",
    )
    new.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    new.add_argument(
        "--reranker", default="off_the_shelf",
        choices=["off_the_shelf", "skip"],
        help="reranker strategy (slice 14.0 supports off_the_shelf only)",
    )
    new.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    # Slice 14.1 / 14.2 knobs.
    new.add_argument(
        "--schema-strategy", default="infer", choices=["infer", "llm"],
        help="how extract-schema generates schema.py",
    )
    new.add_argument(
        "--queries-per-product", type=int, default=10,
        help="how many synthetic queries to generate per product",
    )
    new.add_argument(
        "--hard-negatives-per-query", type=int, default=5,
        help="k for the Rust hard-negative miner",
    )
    new.add_argument(
        "--synthetic-dry-run", action="store_true",
        help="generate templated queries without an LLM (CI / no-API-key mode)",
    )
    new.add_argument(
        "--training-dry-run", action="store_true",
        help="finetune steps emit stub adapters and update manifests, no GPU work",
    )
    new.add_argument(
        "--listwise-base-model", default=None,
        help="base model for the listwise LoRA (default: Qwen/Qwen3-1.7B)",
    )
    new.set_defaults(func=cmd_new)

    # `arena inspect`
    inspect = sub.add_parser("inspect", help="show a bundle's manifest")
    inspect.add_argument("path", help="path to a bundle directory")
    inspect.set_defaults(func=cmd_inspect)

    # `arena pack`
    pack = sub.add_parser("pack", help="pack a bundle into a .tar.gz")
    pack.add_argument("path", help="path to a bundle directory")
    pack.add_argument(
        "--out", default=None,
        help="output archive path (default: <domain>.tar.gz)",
    )
    pack.set_defaults(func=cmd_pack)

    # `arena unpack`
    unpack = sub.add_parser("unpack", help="extract a packed bundle .tar.gz")
    unpack.add_argument("archive", help="path to a .tar.gz")
    unpack.add_argument("--into", default="artifacts", help="parent directory")
    unpack.set_defaults(func=cmd_unpack)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
