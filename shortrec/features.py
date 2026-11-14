"""Shared feature engineering for the ranker.

One function builds the ranker's feature matrix for any set of (user, item)
pairs, so training, evaluation and live serving use the IDENTICAL feature
definitions (train/serve parity). All history-derived features come from
train-only `HistoryStats`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HistoryStats

# Historical engagement features computed from the training period only.
# Indices 0 & 2 are the RATES removed in the ablation. (The retriever score is
# deliberately NOT a ranker feature: on real data it leaked train-period
# exposure and hurt held-out AUC, so the ranker uses engagement history only.)
FEATURE_NAMES = [
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
    stats: HistoryStats,
    item_duration: dict[int, float] | None = None,
) -> np.ndarray:
    """Return an (N, len(FEATURE_NAMES)) float32 feature matrix (vectorized).

    Uses train-only aggregates, so this is the correct, leakage-free builder for
    TEST and SERVE rows (whose labels never entered `stats`). Unknown users/items
    fall back to the global engagement rate / zero counts, so cold ids never
    crash inference. For TRAIN rows use `build_train_features_loo` instead.
    """
    item_duration = item_duration or {}
    g = stats.global_rate
    u, it = pd.Series(users), pd.Series(items)
    return np.column_stack([
        u.map(stats.user_rate).fillna(g).to_numpy(),
        np.log1p(u.map(stats.user_count).fillna(0.0).to_numpy()),
        it.map(stats.item_rate).fillna(g).to_numpy(),
        np.log1p(it.map(stats.item_count).fillna(0.0).to_numpy()),
        np.log1p(it.map(item_duration).fillna(0.0).to_numpy()),
    ]).astype(np.float32)


def build_train_features_loo(
    users: np.ndarray,
    items: np.ndarray,
    labels: np.ndarray,
    user_sum: dict[int, float],
    user_count: dict[int, float],
    item_sum: dict[int, float],
    item_count: dict[int, float],
    global_rate: float,
    item_duration: dict[int, float] | None = None,
) -> np.ndarray:
    """Build TRAIN ranker features with **leave-one-out** engagement rates.

    A training row's own label is excluded from its user/item engagement-rate
    features (rate = (sum - label) / (count - 1)); without this, the ranker sees
    its own target leaked through the aggregate and over-fits it. Count features
    use the full exposure count (no label, so not leakage). Combined with the
    chronological split + train-only TEST features, this makes the whole training
    pipeline leakage-safe.
    """
    item_duration = item_duration or {}
    u, it = pd.Series(users), pd.Series(items)
    y = labels.astype(np.float64)

    us = u.map(user_sum).fillna(0.0).to_numpy()
    uc = u.map(user_count).fillna(0.0).to_numpy()
    user_rate = np.where(uc > 1, (us - y) / np.maximum(uc - 1, 1), global_rate)

    isum = it.map(item_sum).fillna(0.0).to_numpy()
    ic = it.map(item_count).fillna(0.0).to_numpy()
    item_rate = np.where(ic > 1, (isum - y) / np.maximum(ic - 1, 1), global_rate)

    return np.column_stack([
        user_rate,
        np.log1p(uc),
        item_rate,
        np.log1p(ic),
        np.log1p(it.map(item_duration).fillna(0.0).to_numpy()),
    ]).astype(np.float32)


def ablation_mask() -> np.ndarray:
    """Boolean mask over FEATURE_NAMES that KEEPS all but the ablated features."""
    return np.array([name not in ABLATION_DROP for name in FEATURE_NAMES])
