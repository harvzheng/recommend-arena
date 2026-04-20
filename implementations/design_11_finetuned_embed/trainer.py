"""Fine-tuning wrapper around sentence-transformers for Design #11."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from .pairs import ContrastivePair, TrainingConfig

logger = logging.getLogger(__name__)


def fine_tune(
    pairs: list[ContrastivePair],
    config: TrainingConfig,
) -> SentenceTransformer:
    """Fine-tune MiniLM on contrastive pairs.

    Uses MultipleNegativesRankingLoss which treats other in-batch positives
    as negatives. With batch_size=16, each example gets 15 implicit negatives.
    """
    model = SentenceTransformer(config.base_model)

    split_idx = int(len(pairs) * (1 - config.validation_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_examples = [
        InputExample(texts=[p.anchor, p.positive]) for p in train_pairs
    ]

    train_loader = DataLoader(
        train_examples,
        batch_size=config.batch_size,
        shuffle=True,
    )

    loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = None
    if len(val_pairs) >= 2:
        val_sentences_1 = [p.anchor for p in val_pairs]
        val_sentences_2 = [p.positive for p in val_pairs]
        val_scores = [1.0] * len(val_pairs)
        evaluator = EmbeddingSimilarityEvaluator(
            val_sentences_1, val_sentences_2, val_scores,
            name="attribute-val",
        )

    warmup_steps = int(len(train_loader) * config.epochs * config.warmup_fraction)

    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=config.epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=len(train_loader) if evaluator else 0,
        output_path=config.output_dir,
        optimizer_params={"lr": config.learning_rate},
        show_progress_bar=False,
    )

    return model


def export_model(model: SentenceTransformer, config: TrainingConfig) -> Path:
    """Save the fine-tuned model for inference."""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model.save(str(output / "model"))

    with open(output / "training_config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    return output
