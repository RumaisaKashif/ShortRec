"""Shared feature engineering for the ranker.

One function builds the ranker's feature matrix for any set of (user, item)
pairs, so training, evaluation and live serving use the IDENTICAL feature
definitions (train/serve parity). All history-derived features come from
train-only `HistoryStats`.
"""
from __future__ import annotations

import numpy as np

from .data import HistoryStats

# Order matters: index 0 is the retrieval score; indices 1 & 3 are the
# historical engagement RATES removed in the ablation.
FEATURE_NAMES = [
    "retriever_score",
    "user_hist_rate",
    "user_hist_count_log",
    "item_hist_rate",
    "item_hist_count_log",
    "item_duration_log",
]
ABLATION_DROP = ("user_hist_rate", "item_hist_rate")


def build_features(
    users: np.ndarray,
    items: np.ndarray,
    retriever_scores: np.ndarray,
    stats: HistoryStats,
    item_duration: dict[int, float] | None = None,
) -> np.ndarray:
    """Return an (N, len(FEATURE_NAMES)) float32 feature matrix.

    Unknown users/items fall back to the global engagement rate / zero counts,
    so cold ids never crash inference.
    """
    item_duration = item_duration or {}
    n = len(users)
    feats = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    g = stats.global_rate
    for row in range(n):
        u = int(users[row])
        it = int(items[row])
        feats[row, 0] = retriever_scores[row]
        feats[row, 1] = stats.user_rate.get(u, g)
        feats[row, 2] = np.log1p(stats.user_count.get(u, 0.0))
        feats[row, 3] = stats.item_rate.get(it, g)
        feats[row, 4] = np.log1p(stats.item_count.get(it, 0.0))
        feats[row, 5] = np.log1p(item_duration.get(it, 0.0))
    return feats


def ablation_mask() -> np.ndarray:
    """Boolean mask over FEATURE_NAMES that KEEPS all but the ablated features."""
    return np.array([name not in ABLATION_DROP for name in FEATURE_NAMES])
