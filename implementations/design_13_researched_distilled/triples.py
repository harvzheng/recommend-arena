"""Score-derived contrastive triples for retrieval training (spec §6.1).

Each triple = (query, positive_passage, negative_passage, score_margin).
Margin is used by the loss to weight high-confidence triples more.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from implementations.design_13_researched_distilled.teacher import TeacherJudgment


@dataclass
class TrainingTriple:
    query: str
    positive_passage: str
    negative_passage: str
    score_margin: float


@dataclass
class _ProductPassages:
    passages: list[str]


def build_triples(
    judgments_by_query: dict[str, list[TeacherJudgment]],
    products: dict[str, _ProductPassages],
    query_text_by_id: dict[str, str],
    pos_threshold: float = 0.7,
    neg_threshold: float = 0.3,
    seed: int = 42,
) -> list[TrainingTriple]:
    rng = random.Random(seed)
    out: list[TrainingTriple] = []
    for query_id, js in judgments_by_query.items():
        query_text = query_text_by_id.get(query_id)
        if not query_text:
            continue
        scores = sorted(js, key=lambda j: j.score)
        positives = [j for j in scores if j.score >= pos_threshold]
        negatives = [j for j in scores if j.score <= neg_threshold]
        if len(positives) < 2 or len(negatives) < 2:
            n = len(scores)
            if n < 4:
                continue
            negatives = scores[: n // 4]
            positives = scores[-(n // 4) :]
        for pos in positives:
            pos_passages = products.get(pos.product_id, _ProductPassages([])).passages
            if not pos_passages:
                continue
            for neg in negatives:
                neg_passages = products.get(neg.product_id, _ProductPassages([])).passages
                if not neg_passages:
                    continue
                out.append(TrainingTriple(
                    query=query_text,
                    positive_passage=rng.choice(pos_passages),
                    negative_passage=rng.choice(neg_passages),
                    score_margin=pos.score - neg.score,
                ))
    return out
