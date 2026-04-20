"""LoRA training via transformers + peft (Apple Silicon compatible)."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import torch

from .db import insert_training_run

logger = logging.getLogger(__name__)

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _get_device() -> str:
    """Detect the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_lora_config(rank: int = 16, alpha: int = 32, dropout: float = 0.05) -> dict:
    return {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": list(TARGET_MODULES),
        "lora_dropout": dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def _build_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1024,
) -> dict:
    device = _get_device()
    # MPS doesn't support fp16 training well; use bf16 on MPS/CUDA, fp32 on CPU
    use_bf16 = device in ("mps", "cuda")
    return {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "bf16": use_bf16,
        "fp16": False,
        "optim": "adamw_torch",
        "dataloader_pin_memory": device != "mps",
        "use_mps_device": device == "mps",
        # SFT-specific (trl >= 0.24 moved these into SFTConfig)
        "dataset_text_field": "text",
        "max_length": max_seq_length,
    }


def _format_example(example: dict) -> dict:
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": prompt}


def train_student(
    dataset_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "adapters/design-12",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    db: sqlite3.Connection | None = None,
) -> str:
    """Fine-tune Qwen 2.5 on teacher judgments via LoRA.

    Runs on Apple Silicon (MPS), CUDA, or CPU. No unsloth dependency.
    Requires: peft, transformers, trl, datasets, torch.
    """
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    device = _get_device()
    logger.info(
        "Starting training: base=%s, dataset=%s, epochs=%d, rank=%d, device=%s",
        base_model, dataset_path, num_epochs, lora_rank, device,
    )

    # Load model — float32 for MPS compatibility, bf16 for CUDA
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device} if device != "cpu" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(_format_example)

    num_examples = len(dataset)

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            **_build_training_args(output_dir, num_epochs, batch_size, learning_rate, max_seq_length)
        ),
    )

    train_result = trainer.train()
    final_loss = train_result.training_loss

    # Save adapter
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete. Adapter saved to %s. Final loss: %.4f", output_dir, final_loss)

    if db is not None:
        insert_training_run(
            db,
            base_model=base_model,
            lora_rank=lora_rank,
            num_examples=num_examples,
            num_epochs=num_epochs,
            final_loss=final_loss,
            adapter_path=output_dir,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        db.commit()

    return output_dir
