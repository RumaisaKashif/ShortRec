"""Data loading, leakage-safe temporal splitting, and feature engineering.

Works on the KuaiRand-Pure interaction log (or the synthetic stand-in, which
mirrors its schema). All historical statistics used as features are computed
**from the training period only**, so no future information leaks into the
train/validation/test evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Loading + splitting
# ----------------------------------------------------------------------------
def load_interactions(path: str, cfg: dict) -> pd.DataFrame:
    """Load an interaction log and standardise columns to:
    user_id, item_id, ts, click, engage.

    `engage` (default: KuaiRand's `long_view`) is the watch-through label used
    both as the retrieval positive signal and the ranker target.
    """
    df = pd.read_csv(path)
    rename = {
        cfg["user_id_col"]: "user_id",
        cfg["item_id_col"]: "item_id",
        cfg["time_col"]: "ts",
        cfg["click_col"]: "click",
        cfg["engage_col"]: "engage",
    }
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in {path}. Columns are: {list(df.columns)}")

    keep = ["user_id", "item_id", "ts", "click", "engage"]
    duration_col = cfg.get("duration_col")
    if duration_col and duration_col in df.columns:
        df = df.rename(columns={**rename, duration_col: "duration"})
        keep.append("duration")
    else:
        df = df.rename(columns=rename)
    df = df[keep].copy()

    df = df.dropna(subset=["user_id", "item_id", "ts"])
    df["user_id"] = df["user_id"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.int64)
    df["click"] = df["click"].fillna(0).astype(np.int8)
    df["engage"] = df["engage"].fillna(0).astype(np.int8)
    if "duration" in df.columns:
        df["duration"] = df["duration"].fillna(0).astype(np.int64)
    return df.sort_values("ts", kind="mergesort").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Chronological split by interaction time (no shuffling, no leakage)."""
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return (
        df.iloc[:t1].copy().reset_index(drop=True),
        df.iloc[t1:t2].copy().reset_index(drop=True),
        df.iloc[t2:].copy().reset_index(drop=True),
    )


# ----------------------------------------------------------------------------
# ID vocabularies
# ----------------------------------------------------------------------------
@dataclass
class IdMaps:
    user2idx: dict[int, int]
    item2idx: dict[int, int]

    @property
    def n_users(self) -> int:
        return len(self.user2idx)

    @property
    def n_items(self) -> int:
        return len(self.item2idx)

    def map_users(self, s: pd.Series) -> np.ndarray:
        return s.map(self.user2idx).to_numpy(dtype=np.int64, copy=True)

    def map_items(self, s: pd.Series) -> np.ndarray:
        return s.map(self.item2idx).to_numpy(dtype=np.int64, copy=True)


def build_id_maps(df: pd.DataFrame) -> IdMaps:
    """Build contiguous index vocabularies over all observed ids.

    (Indexing the full id space is not label leakage — only historical
    *statistics* must be train-only, which `build_history_stats` enforces.)
    """
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    return IdMaps(
        user2idx={u: i for i, u in enumerate(users)},
        item2idx={v: i for i, v in enumerate(items)},
    )


# ----------------------------------------------------------------------------
# Train-only history statistics (features)
# ----------------------------------------------------------------------------
@dataclass
class HistoryStats:
    user_rate: dict[int, float]
    user_count: dict[int, float]
    item_rate: dict[int, float]
    item_count: dict[int, float]
    global_rate: float


def build_history_stats(train_df: pd.DataFrame) -> HistoryStats:
    """Engagement rates/counts computed strictly from the training period."""
    ug = train_df.groupby("user_id")["engage"]
    ig = train_df.groupby("item_id")["engage"]
    return HistoryStats(
        user_rate=ug.mean().to_dict(),
        user_count=ug.count().to_dict(),
        item_rate=ig.mean().to_dict(),
        item_count=ig.count().to_dict(),
        global_rate=float(train_df["engage"].mean()),
    )


# ----------------------------------------------------------------------------
# Exposed-negative sampling for two-tower training
# ----------------------------------------------------------------------------
def build_exposed_negatives(train_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """user_id -> array of items the user was SHOWN but did not engage with.

    Using exposed (impression) negatives rather than random unseen items avoids
    assuming every un-watched video is irrelevant — the user may simply never
    have been shown it. This is the more defensible negative signal on logged
    feedback data.
    """
    neg = train_df.loc[train_df["engage"] == 0]
    return {u: g["item_id"].to_numpy() for u, g in neg.groupby("user_id")}


def positives(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that count as a positive (engaged) interaction."""
    return df.loc[df["engage"] == 1, ["user_id", "item_id"]].reset_index(drop=True)


def user_seen_items(df: pd.DataFrame) -> dict[int, set]:
    """user_id -> set of items interacted with (to exclude from retrieval)."""
    return {u: set(g["item_id"]) for u, g in df.groupby("user_id")}
