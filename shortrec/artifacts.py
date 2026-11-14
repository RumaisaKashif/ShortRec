"""Save / load trained artifacts so the demo and any server can reuse them."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .data import HistoryStats, IdMaps
from .models import MLPRanker, TwoTower


def save_bundle(
    out_dir: str,
    two_tower: TwoTower,
    ranker: MLPRanker,
    scaler,
    id_maps: IdMaps,
    stats: HistoryStats,
    item_duration: dict,
    meta: dict,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(two_tower.state_dict(), os.path.join(out_dir, "two_tower.pt"))
    torch.save(ranker.state_dict(), os.path.join(out_dir, "ranker.pt"))
    np.savez(
        os.path.join(out_dir, "scaler.npz"),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )
    _dump_json(os.path.join(out_dir, "id_maps.json"), {
        "user2idx": {str(k): v for k, v in id_maps.user2idx.items()},
        "item2idx": {str(k): v for k, v in id_maps.item2idx.items()},
    })
    _dump_json(os.path.join(out_dir, "stats.json"), {
        "user_rate": {str(k): v for k, v in stats.user_rate.items()},
        "user_count": {str(k): v for k, v in stats.user_count.items()},
        "item_rate": {str(k): v for k, v in stats.item_rate.items()},
        "item_count": {str(k): v for k, v in stats.item_count.items()},
        "global_rate": stats.global_rate,
    })
    _dump_json(os.path.join(out_dir, "item_duration.json"), {str(k): v for k, v in item_duration.items()})
    _dump_json(os.path.join(out_dir, "meta.json"), meta)


def load_bundle(in_dir: str, device) -> dict:
    meta = _load_json(os.path.join(in_dir, "meta.json"))
    ids = _load_json(os.path.join(in_dir, "id_maps.json"))
    id_maps = IdMaps(
        user2idx={int(k): v for k, v in ids["user2idx"].items()},
        item2idx={int(k): v for k, v in ids["item2idx"].items()},
    )
    two_tower = TwoTower(id_maps.n_users, id_maps.n_items, meta["emb_dim"]).to(device)
    two_tower.load_state_dict(torch.load(os.path.join(in_dir, "two_tower.pt"), map_location=device))
    two_tower.eval()

    ranker = MLPRanker(meta["ranker_in_dim"], tuple(meta["ranker_hidden_dims"])).to(device)
    ranker.load_state_dict(torch.load(os.path.join(in_dir, "ranker.pt"), map_location=device))
    ranker.eval()

    sc = np.load(os.path.join(in_dir, "scaler.npz"))
    s = _load_json(os.path.join(in_dir, "stats.json"))
    stats = HistoryStats(
        user_rate={int(k): v for k, v in s["user_rate"].items()},
        user_count={int(k): v for k, v in s["user_count"].items()},
        item_rate={int(k): v for k, v in s["item_rate"].items()},
        item_count={int(k): v for k, v in s["item_count"].items()},
        global_rate=s["global_rate"],
    )
    item_duration = {int(k): v for k, v in _load_json(os.path.join(in_dir, "item_duration.json")).items()}
    return {
        "two_tower": two_tower, "ranker": ranker,
        "scaler_mean": sc["mean"], "scaler_scale": sc["scale"],
        "id_maps": id_maps, "stats": stats,
        "item_duration": item_duration, "meta": meta,
    }


def _dump_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)
