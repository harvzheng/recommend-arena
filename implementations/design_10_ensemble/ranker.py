"""XGBoost Learning-to-Rank training and inference.

Trains an XGBRanker with pairwise objective on synthetic judgment data,
provides sigmoid-normalized scoring and feature-importance-based explanations.
"""

from __future__ import annotations

import logging

import numpy as np
import xgboost as xgb

from .features import FEATURE_NAMES

logger = logging.getLogger(__name__)


class LTRRanker:
    """XGBoost-based Learning to Rank model."""

    def __init__(self) -> None:
        self.model: xgb.XGBRanker | None = None
        self.feature_importances_: np.ndarray | None = None

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        query_ids: np.ndarray,
    ) -> None:
        """Train the LTR ranker.

        Args:
            features: (n_samples, n_features) feature matrix.
            labels: (n_samples,) relevance labels (0-3).
            query_ids: (n_samples,) query group identifiers.
        """
        if len(features) == 0:
            logger.warning("No training data provided, using fallback ranker")
            return

        # Compute group sizes from query_ids
        unique_qids, group_sizes = np.unique(query_ids, return_counts=True)

        # Filter out groups with only 1 sample (XGBoost pairwise needs pairs)
        valid_mask = np.zeros(len(features), dtype=bool)
        for qid, size in zip(unique_qids, group_sizes):
            if size >= 2:
                valid_mask |= (query_ids == qid)

        if valid_mask.sum() < 4:
            logger.warning(
                "Not enough training pairs (need >= 2 per query group), "
                "using fallback ranker"
            )
            return

        features = features[valid_mask]
        labels = labels[valid_mask]
        query_ids = query_ids[valid_mask]

        # Recompute group sizes after filtering
        unique_qids, group_sizes = np.unique(query_ids, return_counts=True)

        self.model = xgb.XGBRanker(
            objective="rank:pairwise",
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            tree_method="hist",
            random_state=42,
        )

        self.model.fit(
            features,
            labels,
            group=group_sizes.tolist(),
            verbose=False,
        )

        self.feature_importances_ = self.model.feature_importances_
        logger.info(
            "LTR ranker trained on %d samples across %d queries",
            len(features),
            len(unique_qids),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict relevance scores for feature matrix.

        Returns raw (unbounded) scores. Use normalize_scores() for [0,1].
        """
        if self.model is None:
            # Fallback: weighted sum of key signals
            # features columns: bm25, vector_sim, attr_match, attr_sent,
            #                   neg_penalty, review_count, avg_rating,
            #                   attr_coverage, hard_filter, sent_gap
            weights = np.array(
                [0.25, 0.25, 0.20, 0.10, -0.30, 0.02, 0.05, 0.05, 0.10, -0.05],
                dtype=np.float32,
            )
            return features @ weights

        return self.model.predict(features)

    def predict_contributions(self, features: np.ndarray) -> np.ndarray:
        """Get per-feature SHAP contributions for explanations.

        Returns:
            ndarray of shape (n_samples, n_features + 1) where the last
            column is the bias term.
        """
        if self.model is None:
            # Fallback: just return features scaled by fallback weights
            weights = np.array(
                [0.25, 0.25, 0.20, 0.10, -0.30, 0.02, 0.05, 0.05, 0.10, -0.05],
                dtype=np.float32,
            )
            contributions = features * weights
            # Add bias column
            bias = np.zeros((len(features), 1), dtype=np.float32)
            return np.hstack([contributions, bias])

        booster = self.model.get_booster()
        dmat = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
        return booster.predict(dmat, pred_contribs=True)


def normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Sigmoid normalization of XGBoost ranker output to 0-1 range."""
    return 1.0 / (1.0 + np.exp(-raw_scores))


def build_explanation(
    shap_row: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Convert SHAP contribution values into a human-readable explanation.

    Args:
        shap_row: Array of per-feature contributions (last element is bias).

    Returns:
        (explanation_string, matched_attributes_dict)
    """
    contributions = sorted(
        zip(FEATURE_NAMES, shap_row[:-1]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    parts = [
        f"{name} ({val:+.2f})" for name, val in contributions if abs(val) > 0.01
    ]
    explanation = "Ranked here because: " + ", ".join(parts[:4]) if parts else "No strong signal"
    matched = {name: float(val) for name, val in contributions if val > 0.01}
    return explanation, matched
