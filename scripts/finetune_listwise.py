#!/usr/bin/env python3
"""finetune_listwise.py — LoRA fine-tune Qwen3-1.7B as a listwise reranker.

Training data: (query, [N candidates], gold_ranking) triples expanded
from the synthetic_train.jsonl in the bundle. For each (query,
positive, hard_negatives) row we build:

    candidates = [positive] + hard_negatives + (random fillers up to N)
    gold_ranking = positive first, then hard_negatives in original order
    (caller supplied them as "hardest first" via cosine ranking)

Loss: standard supervised fine-tuning on the completion. We mask the
prompt tokens; only the ranking line `[k] > [k] > ...` contributes to
the loss. This teaches the model the format and the domain shift.

Hardware: ~1B active params + LoRA → fits MPS in fp16. Training time
~5-10 min on M3 Max for a few hundred triples.

Usage:
    python scripts/finetune_listwise.py --bundle artifacts/ski \\
        --base-model Qwen/Qwen3-1.7B \\
        --epochs 2 --batch-size 1 --grad-accum 8

Output: writes a LoRA adapter to <bundle>/listwise_adapter/ and updates
the manifest's reranker entry to point at it.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.domain_bundle import Bundle, RerankerArtifact  # noqa: E402
from shared.seeding import seed_all  # noqa: E402
from implementations.design_14_local_hybrid import listwise_rerank  # noqa: E402

logger = logging.getLogger("finetune_listwise")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--candidates-per-query", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
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
    seed_all(args.seed)

    bundle = Bundle.load(args.bundle)
    triples_path = bundle.paths.root / "synthetic_train.jsonl"
    if not triples_path.is_file():
        logger.error("missing %s; run generate_synthetic_queries.py first", triples_path)
        return 2

    triples = [json.loads(l) for l in triples_path.open() if l.strip()]
    products = bundle.read_products()
    reviews = bundle.read_reviews()
    products_by_id = {p.get("id") or p.get("product_id"): p for p in products}
    reviews_by_id: dict[str, list[str]] = {}
    for r in reviews:
        pid = r.get("product_id") or r.get("id")
        if pid:
            reviews_by_id.setdefault(pid, []).append(r.get("text", ""))

    all_pids = list(products_by_id.keys())

    def doc_for(pid: str) -> str:
        p = products_by_id.get(pid, {})
        bits = [p.get("name", ""), p.get("brand", ""), p.get("category", "")]
        attrs = p.get("attributes") or {}
        if attrs:
            # Cap to top 6 attributes by key order to keep prompt tight.
            kv = sorted(attrs.items())[:6]
            bits.append(", ".join(f"{k}={v}" for k, v in kv))
        return ". ".join(b for b in bits if b)[:200]

    # Build (query, candidates, gold_order) examples.
    examples = []
    for t in triples:
        pos = t["positive_id"]
        negs = list(t.get("hard_negatives") or [])
        # Pad to candidates_per_query with random other products
        existing = {pos, *negs}
        fillers = [p for p in all_pids if p not in existing]
        random.shuffle(fillers)
        candidates = [pos] + negs + fillers
        candidates = candidates[: args.candidates_per_query]
        # Shuffle the candidate order, but remember where pos and negs landed.
        # The model sees a shuffled list; the gold ranking is pos-first then
        # negs in their cosine-hardness order, fillers last in catalog order.
        order_index = {pid: i for i, pid in enumerate(candidates)}
        random.shuffle(candidates)
        # Gold ordering by original position (pos was 0, then negs, then fillers)
        gold_order = sorted(
            range(len(candidates)),
            key=lambda new_i: order_index[candidates[new_i]],
        )
        examples.append({
            "query": t["query"],
            "candidates": [(pid, doc_for(pid)) for pid in candidates],
            "gold_indices_1based": [g + 1 for g in gold_order],
        })

    logger.info("built %d listwise training examples", len(examples))

    if args.dry_run:
        return _dry_run(bundle, args.base_model, examples, args)

    # ---- Lazy import of the heavy ML deps ---------------------------------
    try:
        import torch  # type: ignore
        from torch.utils.data import Dataset, DataLoader  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup,
        )
        from peft import LoraConfig, get_peft_model  # type: ignore
    except ImportError as e:
        logger.error("listwise fine-tune needs torch + transformers + peft: %s", e)
        return 2

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("device=%s", device)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    logger.info("loading base model %s …", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model = model.to(device)
    model.print_trainable_parameters()

    # ---- Build prompt + completion strings, tokenize ----------------------
    def render(ex: dict) -> tuple[str, str]:
        prompt = listwise_rerank.build_prompt(ex["query"], ex["candidates"])
        completion = " > ".join(f"[{k}]" for k in ex["gold_indices_1based"])
        return prompt, completion

    class _DS(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            prompt, completion = render(examples[i])
            full = prompt + completion + tok.eos_token
            enc = tok(
                full, truncation=True, max_length=args.max_seq_length,
                return_tensors="pt", add_special_tokens=False,
            )
            input_ids = enc["input_ids"][0]
            # Mask prompt tokens out of the loss
            prompt_ids = tok(
                prompt, truncation=True, max_length=args.max_seq_length,
                return_tensors="pt", add_special_tokens=False,
            )["input_ids"][0]
            labels = input_ids.clone()
            labels[: len(prompt_ids)] = -100
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": enc["attention_mask"][0],
            }

    def _collate(batch):
        # Right-pad to longest in batch
        max_len = max(b["input_ids"].size(0) for b in batch)
        ids = torch.full((len(batch), max_len), tok.pad_token_id, dtype=torch.long)
        labs = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["input_ids"].size(0)
            ids[i, :n] = b["input_ids"]
            labs[i, :n] = b["labels"]
            attn[i, :n] = b["attention_mask"]
        return {"input_ids": ids, "labels": labs, "attention_mask": attn}

    loader = DataLoader(
        _DS(), batch_size=args.batch_size, shuffle=True, collate_fn=_collate,
    )
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    logger.info(
        "training: epochs=%d batch=%d grad_accum=%d total_steps=%d",
        args.epochs, args.batch_size, args.grad_accum, total_steps,
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=max(1, total_steps),
    )

    model.train()
    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()

            if (i + 1) % args.grad_accum == 0 or i == len(loader) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                scheduler.step()
                optim.zero_grad()
                step += 1

            if (i + 1) % 25 == 0:
                logger.info(
                    "epoch=%d step=%d batch=%d/%d loss=%.4f",
                    epoch, step, i + 1, len(loader), running / (i + 1),
                )
        logger.info("epoch %d done, mean loss = %.4f", epoch, running / max(1, len(loader)))

    # ---- Save adapter -----------------------------------------------------
    out_dir = bundle.paths.root / "listwise_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    logger.info("saved listwise adapter to %s", out_dir)

    bundle.manifest.reranker = RerankerArtifact(
        kind="lora",
        base_model=args.base_model,
        adapter_path=out_dir.name,
    )
    bundle.manifest.metadata.setdefault("reranker_runtime_kind", "listwise")
    bundle.manifest.metadata["reranker_runtime_kind"] = "listwise"
    bundle.manifest.metadata.setdefault("training", {})
    bundle.manifest.metadata["training"]["listwise"] = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "n_examples": len(examples),
        "candidates_per_query": args.candidates_per_query,
        "seed": args.seed,
    }
    bundle.save_manifest()
    logger.info("manifest updated.")
    return 0


def _dry_run(bundle: Bundle, base_model: str, examples: list[dict], args) -> int:
    """No-GPU smoke test: stub adapter + manifest update so arena_new
    can exercise the full pipeline without downloading a 1.7B model."""
    out_dir = bundle.paths.root / "listwise_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    stub = out_dir / "adapter_model.safetensors.stub"
    stub.write_text(json.dumps({
        "_dry_run": True,
        "base_model": base_model,
        "n_examples": len(examples),
    }) + "\n")
    bundle.manifest.reranker = RerankerArtifact(
        kind="lora", base_model=base_model, adapter_path=out_dir.name,
    )
    bundle.manifest.metadata["reranker_runtime_kind"] = "listwise"
    bundle.manifest.metadata.setdefault("training", {})
    bundle.manifest.metadata["training"]["listwise"] = {
        "_dry_run": True,
        "base_model": base_model,
        "n_examples": len(examples),
        "candidates_per_query": args.candidates_per_query,
        "lora_rank": args.lora_rank,
        "seed": args.seed,
    }
    bundle.save_manifest()
    logger.info("dry-run: wrote stub %s and updated manifest", stub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
