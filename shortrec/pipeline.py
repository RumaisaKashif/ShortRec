"""End-to-end pipeline: data -> retriever -> ranker -> evaluation -> artifacts."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import data as D
from .artifacts import save_bundle
from .features import ablation_mask, build_features, build_train_features_loo
from .ranking import evaluate_ranker, logreg_baseline, train_ranker
from .retrieval import evaluate_retrieval, popularity_recall, train_two_tower
from .utils import get_device, set_seed


def _item_durations(df: pd.DataFrame) -> dict:
    """Static per-item duration (not leakage); empty if the column is absent."""
    if "duration" in df.columns:
        return df.groupby("item_id")["duration"].first().to_dict()
    return {}


def run(cfg: dict, df: pd.DataFrame | None = None, save: bool = True) -> dict:
    """Run the full pipeline and return a metrics dict.

    `df` lets callers inject data (e.g. synthetic); otherwise it is loaded from
    `cfg['data']['interactions_path']`.
    """
    set_seed(cfg["seed"])
    device = get_device()
    rng = np.random.default_rng(cfg["seed"])

    if df is None:
        df = D.load_interactions(cfg["data"]["interactions_path"], cfg["data"])

    durations = _item_durations(df)

    train_df, _val_df, test_df = D.temporal_split(df, cfg["data"]["train_frac"], cfg["data"]["val_frac"])
    id_maps = D.build_id_maps(df)
    stats = D.build_history_stats(train_df)
    seen = D.user_seen_items(train_df)

    # ---- Stage 1: retrieval ----
    rcfg = cfg["retriever"]
    retriever = train_two_tower(D.positives(train_df), D.build_exposed_negatives(train_df),
                                id_maps, rcfg, device, rng)
    test_pos = D.positives(test_df)
    recall = evaluate_retrieval(retriever, test_pos, id_maps, seen, rcfg["recall_k"], device)
    pop_recall = popularity_recall(train_df, test_pos, seen, rcfg["recall_k"])

    # ---- Stage 2: ranking ----
    # TRAIN features use leave-one-out engagement rates (a row never sees its own
    # label); TEST features use train-only aggregates (no future leakage into the
    # held-out metric).
    usum = train_df.groupby("user_id")["engage"].sum().to_dict()
    ucnt = train_df.groupby("user_id")["engage"].count().to_dict()
    isum = train_df.groupby("item_id")["engage"].sum().to_dict()
    icnt = train_df.groupby("item_id")["engage"].count().to_dict()
    ytr = train_df["engage"].to_numpy()
    Xtr = build_train_features_loo(
        train_df["user_id"].to_numpy(), train_df["item_id"].to_numpy(), ytr,
        usum, ucnt, isum, icnt, stats.global_rate, durations,
    )
    Xte = build_features(test_df["user_id"].to_numpy(), test_df["item_id"].to_numpy(),
                         stats, durations)
    yte = test_df["engage"].to_numpy()
    users_te = test_df["user_id"].to_numpy()

    ranker, scaler = train_ranker(Xtr, ytr, cfg["ranker"], device)
    rank_metrics = evaluate_ranker(ranker, scaler, Xte, yte, users_te, device)
    lr_metrics = logreg_baseline(Xtr, ytr, Xte, yte, users_te)

    # ---- Ablation: drop historical engagement-rate features ----
    mask = ablation_mask()
    abl_ranker, abl_scaler = train_ranker(Xtr[:, mask], ytr, cfg["ranker"], device)
    abl_metrics = evaluate_ranker(abl_ranker, abl_scaler, Xte[:, mask], yte, users_te, device)

    metrics = {
        "n_interactions": int(len(df)),
        "n_users": id_maps.n_users,
        "n_items": id_maps.n_items,
        "retrieval": {
            f"recall@{rcfg['recall_k']}": round(recall, 4),
            f"popularity_recall@{rcfg['recall_k']}": round(pop_recall, 4),
            "lift_vs_popularity": round(recall / pop_recall, 2) if pop_recall > 0 else None,
        },
        "ranking": {
            "mlp_auc": _r(rank_metrics["auc"]),
            "mlp_gauc": _r(rank_metrics["gauc"]),
            "logreg_auc": _r(lr_metrics["auc"]),
            "logreg_gauc": _r(lr_metrics["gauc"]),
            "gauc_lift_vs_logreg": _r(rank_metrics["gauc"] - lr_metrics["gauc"]),
        },
        "ablation_no_hist_rates": {
            "mlp_gauc": _r(abl_metrics["gauc"]),
            "gauc_drop": _r(rank_metrics["gauc"] - abl_metrics["gauc"]),
        },
    }

    if save:
        out = cfg["artifacts_dir"]
        save_bundle(
            out, retriever, ranker, scaler, id_maps, stats, durations,
            meta={
                "emb_dim": rcfg["emb_dim"],
                "ranker_in_dim": Xtr.shape[1],
                "ranker_hidden_dims": list(cfg["ranker"]["hidden_dims"]),
                "recall_k": rcfg["recall_k"],
                "metrics": metrics,
            },
        )
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
