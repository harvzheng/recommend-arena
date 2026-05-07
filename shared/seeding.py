"""Single place that seeds every RNG the trainers touch.

Same recipe → same ballpark. Bit-for-bit determinism on MPS isn't
achievable for the ops we use (sentence-transformers `model.fit`,
HF `model.generate`); this helper just removes the avoidable drift
sources so re-runs land within ±0.02 NDCG instead of ±0.05.
"""

from __future__ import annotations

import os
import random


def seed_all(seed: int) -> None:
    """Seed python.random, numpy, torch (CPU + accelerator)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass
