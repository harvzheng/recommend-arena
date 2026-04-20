"""
benchmark/metrics.py — Metric computation functions for the recommendation benchmark.
"""

from __future__ import annotations

import math


def compute_ndcg(predicted_ids: list[str], gt_relevance: dict[str, int], k: int) -> float:
    """
    Compute NDCG@k.

    Args:
        predicted_ids: Ordered list of product IDs returned by the system
        gt_relevance: {product_id: relevance_grade} from ground truth (0-3 scale)
        k: Cutoff position

    Returns:
        NDCG@k score in [0, 1]
    """
    def dcg(relevances: list[float], k: int) -> float:
        return sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(relevances[:k])
        )

    # Actual DCG from predicted ranking
    actual_rels = [gt_relevance.get(pid, 0) for pid in predicted_ids]
    actual_dcg = dcg(actual_rels, k)

    # Ideal DCG from perfect ranking
    ideal_rels = sorted(gt_relevance.values(), reverse=True)
    ideal_dcg = dcg(ideal_rels, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_mrr(
    predicted_ids: list[str],
    gt_relevance: dict[str, int],
    threshold: int = 3,
) -> float:
    """
    Compute Reciprocal Rank for a single query.

    Returns 1/rank of the first result with relevance >= threshold.
    Returns 0 if no such result exists.
    """
    for i, pid in enumerate(predicted_ids):
        if gt_relevance.get(pid, 0) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def compute_attribute_precision(
    result_dicts: list[dict],
    expected_attributes: list[str],
) -> tuple[float, float, float]:
    """
    Compute attribute precision, recall, and F1.

    Compares the keys in each result's matched_attributes against the
    expected key_attributes for the query.

    Returns:
        (precision, recall, f1) averaged across all results
    """
    if not expected_attributes or not result_dicts:
        return (0.0, 0.0, 0.0)

    expected_set = set(attr.lower() for attr in expected_attributes)
    precisions = []
    recalls = []

    for result in result_dicts:
        matched = result.get("matched_attributes", {})
        if not matched:
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        predicted_set = set(k.lower() for k in matched.keys())
        intersection = predicted_set & expected_set

        p = len(intersection) / len(predicted_set) if predicted_set else 0.0
        r = len(intersection) / len(expected_set) if expected_set else 0.0
        precisions.append(p)
        recalls.append(r)

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    return (avg_p, avg_r, f1)


def compute_coverage(query_results) -> float:
    """
    Fraction of queries with at least one result with relevance >= 2.
    """
    covered = 0
    for qr in query_results:
        gt_map = {item["product_id"]: item["relevance"] for item in qr.ground_truth}
        for result in qr.results:
            if gt_map.get(result["product_id"], 0) >= 2:
                covered += 1
                break
    return covered / len(query_results) if query_results else 0.0


def compute_explanation_quality_proxy(
    result_dicts: list[dict],
    products: list[dict],
) -> float:
    """
    Automated proxy for explanation quality (1-5 scale).

    Checks whether explanations reference actual product attribute names.
    This is a heuristic — full evaluation requires human review.

    Scoring:
        5: mentions 3+ real attributes by name
        4: mentions 2 real attributes
        3: mentions 1 real attribute
        2: non-empty but no attribute match
        1: empty or missing
    """
    # Build attribute name lookup
    attribute_names = set()
    for product in products:
        for attr_name in product.get("attributes", {}).keys():
            attribute_names.add(attr_name.lower().replace("_", " "))
            attribute_names.add(attr_name.lower())

    # Common synonyms
    synonyms = {
        "stiffness": ["stiff", "stiffness", "flex"],
        "damp": ["damp", "dampness", "damping", "vibration"],
        "edge_grip": ["edge grip", "edge hold", "grip", "hold on ice"],
        "stability_at_speed": ["stability", "stable", "speed stability"],
        "playfulness": ["playful", "playfulness", "fun", "lively"],
        "powder_float": ["float", "powder", "flotation"],
        "forgiveness": ["forgiving", "forgiveness", "easy", "beginner-friendly"],
    }
    all_terms = set()
    for terms in synonyms.values():
        all_terms.update(terms)
    all_terms.update(attribute_names)

    scores = []
    for result in result_dicts:
        explanation = (result.get("explanation") or "").lower()
        if not explanation:
            scores.append(1.0)
            continue

        matches = sum(1 for term in all_terms if term in explanation)
        if matches >= 3:
            scores.append(5.0)
        elif matches >= 2:
            scores.append(4.0)
        elif matches >= 1:
            scores.append(3.0)
        else:
            scores.append(2.0)

    return sum(scores) / len(scores) if scores else 1.0
