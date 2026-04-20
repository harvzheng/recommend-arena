"""Tests for trainer.py — fine-tuning on tiny synthetic data."""

import shutil
import tempfile

import numpy as np

from implementations.design_11_finetuned_embed.pairs import (
    ContrastivePair,
    TrainingConfig,
)
from implementations.design_11_finetuned_embed.trainer import export_model, fine_tune


def _make_tiny_pairs() -> list[ContrastivePair]:
    """5 contrastive pairs: enough to verify training runs."""
    return [
        ContrastivePair(anchor="very stiff ski", positive="Incredibly rigid and demanding on hardpack."),
        ContrastivePair(anchor="ski with high stiffness", positive="The stiffest ski I have ever used."),
        ContrastivePair(anchor="playful and fun ski", positive="Super fun to throw around in the park."),
        ContrastivePair(anchor="lively and energetic ski", positive="Bouncy and lively, great for tricks."),
        ContrastivePair(anchor="very stiff ski", positive="Firm underfoot with zero give."),
    ]


def test_fine_tune_runs_and_produces_model():
    tmpdir = tempfile.mkdtemp(prefix="design11_test_")
    try:
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=2e-5,
            validation_split=0.2,
            output_dir=tmpdir,
            seed=42,
        )
        pairs = _make_tiny_pairs()
        model = fine_tune(pairs, config)

        # Model should be able to encode text
        vec = model.encode("stiff ski", normalize_embeddings=True)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        # L2-normalized vector should have norm ~1
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_export_model_saves_files():
    tmpdir = tempfile.mkdtemp(prefix="design11_test_export_")
    try:
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            output_dir=tmpdir,
        )
        pairs = _make_tiny_pairs()
        model = fine_tune(pairs, config)
        output_path = export_model(model, config)

        assert (output_path / "model").exists()
        assert (output_path / "training_config.json").exists()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
