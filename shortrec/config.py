"""Configuration defaults and loader (deep-merges a YAML file over defaults)."""
from __future__ import annotations

import copy

import yaml

DEFAULTS: dict = {
    "seed": 42,
    "data": {
        "interactions_path": "data/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv",
        "user_id_col": "user_id",
        "item_id_col": "video_id",
        "time_col": "time_ms",
        "click_col": "is_click",
        "engage_col": "long_view",
        "duration_col": "duration_ms",  # optional; used as a ranker feature if present
        "train_frac": 0.7,
        "val_frac": 0.1,
    },
    "retriever": {
        "emb_dim": 64,
        "epochs": 5,
        "batch_size": 1024,
        "lr": 1e-3,
        "num_negatives": 4,
        "recall_k": 50,
    },
    "ranker": {
        "hidden_dims": [128, 64],
        "dropout": 0.1,
        "epochs": 5,
        "batch_size": 1024,
        "lr": 1e-3,
    },
    "artifacts_dir": "artifacts",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | None = None) -> dict:
    """Return DEFAULTS, deep-merged with the YAML at `path` if given."""
    if not path:
        return copy.deepcopy(DEFAULTS)
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULTS, user_cfg)
