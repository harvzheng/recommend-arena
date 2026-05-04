#!/usr/bin/env python3
"""finetune_reranker.py — Step 4 of the domain compiler (slice 14.2).

Optional but recommended: train a per-domain LoRA on bge-reranker-v2-m3
to boost the cross-encoder rerank stage.

Pipeline:
  1. Convert synthetic triples (query, positive, hard_negatives[]) into
     pointwise (query, document, label) examples where label ∈ {0, 1}.
  2. LoRA fine-tune the reranker (CrossEncoder).
  3. Save adapter to <bundle>/reranker_lora/.
  4. Update manifest.

Hardware (per the spec): rank-8 LoRA, ~1 hour on 5070 Ti.

Usage:
    python scripts/finetune_reranker.py --bundle artifacts/skis

This script can be skipped entirely; the off-the-shelf reranker is a
strong baseline. Skip when the eval gate already passes without it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.domain_bundle import Bundle, RerankerArtifact  # noqa: E402

logger = logging.getLogger("finetune_reranker")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--base-model", default=None,
        help="override the bundle's manifested reranker base model",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help="comma-separated module names to LoRA-wrap. "
             "Auto-detected if omitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="emit a stub adapter file + update manifest, no actual training",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    bundle = Bundle.load(args.bundle)
    base_model = args.base_model or (
        bundle.manifest.reranker.base_model
        if bundle.manifest.reranker
        else "BAAI/bge-reranker-v2-m3"
    )

    triples_path = bundle.paths.root / "synthetic_train.jsonl"
    if not triples_path.is_file():
        logger.error(
            "%s missing; run scripts/generate_synthetic_queries.py first",
            triples_path,
        )
        return 2

    triples = _read_jsonl(triples_path)
    logger.info("loaded %d triples (base=%s)", len(triples), base_model)

    # Convert triples to pointwise (query, document, label) pairs.
    products_by_id = {
        p.get("id") or p.get("product_id"): p for p in bundle.read_products()
    }

    def _doc_for(pid: str) -> str:
        p = products_by_id.get(pid, {})
        return ". ".join(filter(None, [
            p.get("name") or p.get("product_name") or pid,
            p.get("brand"),
            p.get("category"),
        ]))

    pairs: list[tuple[str, str, int]] = []
    for t in triples:
        q = t["query"]
        pairs.append((q, _doc_for(t["positive_id"]), 1))
        for neg_id in t.get("hard_negatives", []):
            pairs.append((q, _doc_for(neg_id), 0))
    logger.info("expanded to %d pointwise pairs", len(pairs))

    out_dir = bundle.paths.reranker_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return _dry_run(bundle, base_model, pairs, out_dir, args.lora_rank)

    # Real training path.
    try:
        import torch  # type: ignore
        from sentence_transformers import CrossEncoder, InputExample  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
    except ImportError as e:
        logger.error(
            "fine-tune requires torch + sentence-transformers; install with "
            "`pip install -r implementations/design_14_local_hybrid/requirements.txt`"
        )
        logger.error("ImportError: %s", e)
        return 2

    model = CrossEncoder(base_model, num_labels=1, max_length=args.max_seq_length)

    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
        target_modules = _resolve_target_modules(
            args.lora_target_modules, model.model
        )
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=target_modules,
            bias="none",
        )
        model.model = get_peft_model(model.model, lora)
        logger.info(
            "LoRA enabled (rank=%d, target_modules=%s)",
            args.lora_rank, target_modules,
        )
        kind = "lora"
    except ImportError:
        logger.warning("peft not installed; full fine-tune.")
        kind = "full_finetune"

    examples = [InputExample(texts=[q, d], label=float(label)) for q, d, label in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    logger.info(
        "training: epochs=%d batch=%d steps/epoch≈%d",
        args.epochs, args.batch_size, len(loader),
    )
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=int(len(loader) * 0.1),
        optimizer_params={"lr": args.learning_rate},
        show_progress_bar=True,
    )
    model.save(str(out_dir))
    logger.info("saved reranker adapter to %s", out_dir)

    bundle.manifest.reranker = RerankerArtifact(
        kind=kind,
        base_model=base_model,
        adapter_path=str(out_dir.relative_to(bundle.paths.root)),
    )
    bundle.manifest.metadata.setdefault("training", {})
    bundle.manifest.metadata["training"]["reranker"] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "n_pairs": len(pairs),
        "lora_rank": None if kind == "full_finetune" else args.lora_rank,
    }
    bundle.save_manifest()
    logger.info("manifest updated.")
    return 0


def _resolve_target_modules(override: str | None, auto_model) -> list[str]:
    """Pick LoRA target modules. Honors --lora-target-modules; otherwise
    probes the loaded model for a known attention-projection naming.
    """
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    names = {n for n, _ in auto_model.named_modules()}
    if any(n.endswith(".q_proj") for n in names):
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    if any(n.endswith(".query") for n in names):
        return ["query", "key", "value"]
    raise SystemExit(
        "could not auto-detect LoRA target modules; pass --lora-target-modules explicitly"
    )


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _dry_run(bundle: Bundle, base_model: str, pairs: list[tuple[str, str, int]],
             out_dir: Path, lora_rank: int) -> int:
    stub = out_dir / "adapter_model.safetensors.stub"
    stub.write_text(json.dumps({
        "_dry_run": True,
        "base_model": base_model,
        "n_pairs": len(pairs),
    }) + "\n")
    bundle.manifest.reranker = RerankerArtifact(
        kind="lora",
        base_model=base_model,
        adapter_path=str(out_dir.relative_to(bundle.paths.root)),
    )
    bundle.manifest.metadata.setdefault("training", {})
    bundle.manifest.metadata["training"]["reranker"] = {
        "_dry_run": True,
        "n_pairs": len(pairs),
        "lora_rank": lora_rank,
    }
    bundle.save_manifest()
    logger.info(
        "dry-run: wrote stub adapter %s. Re-run without --dry-run on a GPU host.",
        stub,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
