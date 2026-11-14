import numpy as np

from shortrec.metrics import auc, gauc, recall_at_k


def test_recall_at_k_basic():
    ranked = {1: [10, 11, 12], 2: [20, 21, 22]}
    relevant = {1: {11}, 2: {99}}  # user 1 hit, user 2 miss
    assert recall_at_k(ranked, relevant) == 0.5


def test_recall_at_k_partial():
    ranked = {1: [1, 2, 3, 4]}
    relevant = {1: {2, 4, 9}}  # 2 of 3 relevant retrieved
    assert abs(recall_at_k(ranked, relevant) - 2 / 3) < 1e-9


def test_auc_perfect_and_degenerate():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert np.isnan(auc([1, 1, 1], [0.3, 0.6, 0.9]))  # one class -> nan


def test_gauc_skips_single_class_users_and_weights():
    users = np.array([1, 1, 2, 2])
    y = np.array([0, 1, 1, 1])        # user 2 is single-class -> skipped
    score = np.array([0.2, 0.9, 0.5, 0.4])
    assert gauc(users, y, score) == 1.0  # only user 1 scored, perfectly ranked
