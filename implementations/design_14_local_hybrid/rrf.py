"""Thin wrapper over the Rust RRF core, with a pure-Python fallback.

Always prefer the Rust implementation. The fallback exists so the test
suite and offline development can run without rebuilding the wheel
after every checkout. The numerical contract is identical:

    score(d) = sum over retrievers r of  1 / (k + rank_r(d))

with deterministic tie-breaking by first-appearance order.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

try:
    import arena_core  # type: ignore
    _rust_rrf_fuse = getattr(arena_core, "rrf_fuse", None)
    if _rust_rrf_fuse is None:
        # `arena_core` resolved to the Rust source dir as a namespace package
        # (no built extension installed). Fall back to Python.
        raise ImportError("arena_core present but native extension not built")
    _RRF_BACKEND = "rust"
except ImportError:
    _RRF_BACKEND = "python"
    _rust_rrf_fuse = None
    logger.warning(
        "design_14: arena_core (Rust) not installed; using slower Python "
        "RRF fallback. Build with `cd arena_core && maturin develop --release`."
    )


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked ID lists via Reciprocal Rank Fusion."""
    # Normalize inputs to plain lists of strings — the Rust signature is strict.
    normalized = [[str(x) for x in lst] for lst in ranked_lists]

    if _RRF_BACKEND == "rust":
        return _rust_rrf_fuse(normalized, k, top_k)
    return _python_rrf_fallback(normalized, k, top_k)


def _python_rrf_fallback(
    ranked_lists: list[list[str]],
    k: int,
    top_k: int | None,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    counter = 0
    for lst in ranked_lists:
        seen_in_list: set[str] = set()
        for rank0, pid in enumerate(lst):
            if pid in seen_in_list:
                continue
            seen_in_list.add(pid)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank0 + 1.0)
            if pid not in first_seen:
                counter += 1
                first_seen[pid] = counter

    pairs = sorted(
        scores.items(),
        key=lambda x: (-x[1], first_seen[x[0]]),
    )
    if top_k is not None:
        pairs = pairs[:top_k]
    return pairs


def backend_name() -> str:
    """Returns 'rust' if the native extension is loaded, 'python' otherwise."""
    return _RRF_BACKEND
