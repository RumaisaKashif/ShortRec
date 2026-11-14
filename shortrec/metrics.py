"""Evaluation metrics: Recall@K for retrieval, AUC / GAUC for ranking."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def recall_at_k(ranked_items: dict[int, list[int]], relevant: dict[int, set[int]]) -> float:
    """Mean Recall@K over users.

    For each user, Recall = |top-K retrieved ∩ relevant| / |relevant|.
    Only users with >= 1 relevant item are scored.

    Args:
        ranked_items: user_id -> ranked list of candidate item ids (top-K already).
        relevant:     user_id -> set of held-out relevant (engaged) item ids.
    """
    recalls = []
    for user, rel in relevant.items():
        if not rel:
            continue
        topk = ranked_items.get(user, [])
        hits = len(set(topk) & rel)
        recalls.append(hits / len(rel))
    return float(np.mean(recalls)) if recalls else 0.0


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Global ROC-AUC. Returns nan if only one class is present."""
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def gauc(
    user_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    weight_by: str = "impressions",
) -> float:
    """Grouped AUC: per-user AUC averaged, weighted across users.

    GAUC is the standard recsys ranking metric because a global AUC can look
    strong just by separating heavy from light users; GAUC measures whether the
    model orders items correctly *within* each user's own impressions.

    Users whose labels are all-positive or all-negative are skipped (AUC is
    undefined for them). ``weight_by='impressions'`` weights each user by their
    number of impressions; ``'uniform'`` weights every user equally.
    """
    user_ids = np.asarray(user_ids)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    total_w = 0.0
    weighted_sum = 0.0
    for u in np.unique(user_ids):
        mask = user_ids == u
        yt = y_true[mask]
        if yt.min() == yt.max():
            continue
        w = float(mask.sum()) if weight_by == "impressions" else 1.0
        weighted_sum += w * roc_auc_score(yt, y_score[mask])
        total_w += w
    return float(weighted_sum / total_w) if total_w > 0 else float("nan")
