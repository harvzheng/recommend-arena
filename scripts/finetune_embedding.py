#!/usr/bin/env python3
"""finetune_embedding.py — Step 2 of the domain compiler (slice 14.1).

Fine-tune Qwen3-Embedding-0.6B (or any sentence-transformers-compatible
base) on the synthetic triples produced by `generate_synthetic_queries.py`.

Loss: MultipleNegativesRankingLoss with hard negatives. The classic
recipe for retrieval embeddings — pulls `(query, positive)` pairs
together while pushing `(query, hard_negative)` pairs apart.

Output: writes to `<bundle>/embedding/`. Either a LoRA adapter
(default; ~50-150MB) or full weights (--full-finetune; ~1.2GB). Updates
the bundle manifest so design 14's runtime picks up the new model.

Hardware note (per the spec):
- 5070 Ti (16GB Blackwell): batch 16 / seq 256 / ~6GB VRAM, ~30 min for 10K triples.
- M3 Max: ~2–3x slower per step but fits much larger batches.

Usage:
    python scripts/finetune_embedding.py \\
        --bundle artifacts/skis \\
        --epochs 3 \\
        --batch-size 16

This script imports torch / sentence-transformers / peft lazily and
exits with a clear message if they aren't installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.domain_bundle import Bundle, EmbeddingArtifact  # noqa: E402

logger = logging.getLogger("finetune_embedding")


def _check_deps() -> tuple[Any, Any, Any] | None:  # type: ignore[name-defined]
    """Lazy-import the heavy ML deps. Return (torch, ST, losses) or None."""
    try:
        import torch  # type: ignore
        from sentence_transformers import (  # type: ignore
            InputExample, SentenceTransformer, losses,
        )
        return torch, SentenceTransformer, (InputExample, losses)
    except ImportError as e:
        logger.error(
            "fine-tune requires torch + sentence-transformers; install with "
            "`pip install -r implementations/design_14_local_hybrid/requirements.txt`"
        )
        logger.error("ImportError: %s", e)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--base-model", default=None,
        help="override the bundle's manifested base model",
    )
    parser.add_argument(
        "--full-finetune", action="store_true",
        help="train all weights instead of LoRA (default: LoRA)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="don't actually train; just validate the triples + emit a "
             "stub adapter file. Useful in CI to check the pipeline shape.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    bundle = Bundle.load(args.bundle)
    base_model = args.base_model or (
        bundle.manifest.embedding.base_model
        if bundle.manifest.embedding
        else "Qwen/Qwen3-Embedding-0.6B"
    )

    triples_path = bundle.paths.root / "synthetic_train.jsonl"
    if not triples_path.is_file():
        logger.error(
            "%s missing; run scripts/generate_synthetic_queries.py first",
            triples_path,
        )
        return 2

    triples = _read_triples(triples_path)
    logger.info("loaded %d triples (base=%s)", len(triples), base_model)

    out_dir = bundle.paths.embedding_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return _dry_run(args, bundle, base_model, triples, out_dir)

    deps = _check_deps()
    if deps is None:
        return 2
    torch, SentenceTransformer, (InputExample, losses) = deps  # type: ignore

    # Build training examples. MultipleNegativesRankingLoss takes
    # InputExample(texts=[anchor, positive, neg1, neg2, ...]) — extra
    # negatives become "explicit hard negatives" the model must push
    # away from the positive at training time.
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

    train_examples = []
    for t in triples:
        anchor = t["query"]
        positive = _doc_for(t["positive_id"])
        negatives = [_doc_for(n) for n in t.get("hard_negatives", [])]
        train_examples.append(InputExample(texts=[anchor, positive, *negatives]))

    logger.info("loading base model %s …", base_model)
    model = SentenceTransformer(base_model)
    model.max_seq_length = args.max_seq_length

    # LoRA wrap. sentence-transformers exposes the underlying transformer
    # at model[0].auto_model.
    if not args.full_finetune:
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
            lora = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
            )
            model[0].auto_model = get_peft_model(model[0].auto_model, lora)
            logger.info("LoRA enabled (rank=%d)", args.lora_rank)
        except ImportError:
            logger.warning(
                "peft not installed; falling back to full fine-tune. "
                "Install peft to use LoRA."
            )
            args.full_finetune = True

    train_loss = losses.MultipleNegativesRankingLoss(model)

    from torch.utils.data import DataLoader  # type: ignore
    loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    logger.info(
        "training: epochs=%d batch=%d steps/epoch≈%d",
        args.epochs, args.batch_size, len(loader),
    )
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(len(loader) * 0.1),
        optimizer_params={"lr": args.learning_rate},
        show_progress_bar=True,
    )

    model.save(str(out_dir))
    logger.info("saved fine-tuned embedding to %s", out_dir)

    # Update manifest
    bundle.manifest.embedding = EmbeddingArtifact(
        kind="full_finetune" if args.full_finetune else "lora",
        base_model=base_model,
        adapter_path=str(out_dir.relative_to(bundle.paths.root)),
        dim=model.get_sentence_embedding_dimension() or 1024,
    )
    bundle.manifest.metadata.setdefault("training", {})
    bundle.manifest.metadata["training"]["embedding"] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "n_triples": len(triples),
        "lora_rank": None if args.full_finetune else args.lora_rank,
    }
    bundle.save_manifest()
    logger.info("manifest updated.")
    return 0


def _read_triples(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _dry_run(args, bundle: Bundle, base_model: str, triples: list[dict], out_dir: Path) -> int:
    """No-GPU pipeline check: stub an adapter, update manifest."""
    stub = out_dir / "adapter_model.safetensors.stub"
    stub.write_text(
        json.dumps({
            "_dry_run": True,
            "base_model": base_model,
            "n_triples": len(triples),
        }) + "\n"
    )
    bundle.manifest.embedding = EmbeddingArtifact(
        kind="lora",
        base_model=base_model,
        adapter_path=str(out_dir.relative_to(bundle.paths.root)),
        dim=1024,
    )
    bundle.manifest.metadata.setdefault("training", {})
    bundle.manifest.metadata["training"]["embedding"] = {
        "_dry_run": True,
        "n_triples": len(triples),
    }
    bundle.save_manifest()
    logger.info(
        "dry-run: wrote stub adapter %s and updated manifest. "
        "Re-run without --dry-run on a GPU host to actually train.", stub,
    )
    return 0


# ---------------------------------------------------------------------------
# Late-bound for the type hint in _check_deps. Importing Any here avoids
# a top-level dependency the script itself doesn't need.
# ---------------------------------------------------------------------------
from typing import Any  # noqa: E402  (used in _check_deps return annotation)


if __name__ == "__main__":
    raise SystemExit(main())
