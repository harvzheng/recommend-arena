"""Cross-encoder rerank — bge-reranker-v2-m3.

Lazy-loaded; if torch / transformers / FlagEmbedding aren't available
the recommender skips this stage and returns RRF-fused order. That's
still better than #4's score-fusion failure (because RRF is rank-based)
and matches the design 14 fallback strategy.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = os.environ.get(
    "RECOMMEND_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
)

_reranker_cache: dict[str, Any] = {}


def _load_reranker(model_name: str):
    if model_name in _reranker_cache:
        return _reranker_cache[model_name]

    # Try FlagEmbedding first (the official BGE rerank wrapper).
    try:
        from FlagEmbedding import FlagReranker  # type: ignore
        logger.info("design_14: loading reranker %s (FlagEmbedding) …", model_name)
        m = FlagReranker(model_name, use_fp16=True)
        _reranker_cache[model_name] = ("flag", m)
        return _reranker_cache[model_name]
    except ImportError:
        pass

    # Fall back to sentence-transformers CrossEncoder (works with the same model).
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        logger.info(
            "design_14: loading reranker %s (sentence-transformers) …", model_name
        )
        m = CrossEncoder(model_name, max_length=512)
        _reranker_cache[model_name] = ("st", m)
        return _reranker_cache[model_name]
    except ImportError:
        logger.warning(
            "design_14: no reranker library available "
            "(install FlagEmbedding or sentence-transformers); "
            "skipping cross-encoder rerank stage."
        )
        _reranker_cache[model_name] = None
        return None


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_k: int = 10,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[tuple[str, float]]:
    """Rerank (id, document_text) pairs against the query.

    Returns (id, score) sorted descending. Score is the cross-encoder's
    raw relevance score (NOT bounded to [0,1]).

    On failure or missing dependency this returns the input order
    unchanged with score = 1 / (rank+1) — caller can detect that the
    rerank effectively no-op'd by checking whether scores are descending
    geometrically.
    """
    if not candidates:
        return []

    handle = _load_reranker(model_name)
    if handle is None:
        # Pass-through: keep the order RRF gave us.
        return [(pid, 1.0 / (i + 1)) for i, (pid, _doc) in enumerate(candidates)]

    backend, model = handle
    pairs = [(query, doc) for _pid, doc in candidates]
    if backend == "flag":
        scores = model.compute_score(pairs, normalize=False)
    else:
        scores = model.predict(pairs)

    # FlagReranker returns a single float for one pair; wrap in a list.
    if not hasattr(scores, "__iter__"):
        scores = [scores]
    scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
