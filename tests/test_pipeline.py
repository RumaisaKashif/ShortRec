"""Fast integration tests on synthetic data (kept tiny so CI stays quick)."""
import tempfile
from pathlib import Path

import numpy as np

from shortrec import data as D
from shortrec.config import load_config
from shortrec.features import ablation_mask, build_features
from shortrec.pipeline import run
from shortrec.synthetic import make_synthetic_kuairand, write_synthetic


def _tiny_cfg(csv: str, artifacts: str) -> dict:
    cfg = load_config()
    cfg["data"]["interactions_path"] = csv
    cfg["artifacts_dir"] = artifacts
    cfg["retriever"].update(emb_dim=32, epochs=15, batch_size=256, lr=5e-3,
                            num_negatives=8, recall_k=20)
    cfg["ranker"].update(epochs=5, batch_size=256)
    return cfg


def test_temporal_split_is_ordered_and_non_overlapping():
    df = make_synthetic_kuairand(n_interactions=1000, seed=1).rename(
        columns={"video_id": "item_id", "time_ms": "ts", "is_click": "click", "long_view": "engage"}
    )
    df = df.sort_values("ts").reset_index(drop=True)
    tr, va, te = D.temporal_split(df, 0.7, 0.1)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr["ts"].max() <= va["ts"].min() and va["ts"].max() <= te["ts"].min()


def test_ablation_mask_drops_two_features():
    assert ablation_mask().sum() == len(ablation_mask()) - 2


def test_build_features_shape_and_cold_fallback():
    stats = D.HistoryStats({}, {}, {}, {}, global_rate=0.3)
    feats = build_features(np.array([999]), np.array([888]), np.array([0.5]), stats, {})
    assert feats.shape == (1, 6)
    assert feats[0, 1] == 0.3 and feats[0, 3] == 0.3  # unknown user/item -> global rate


def test_pipeline_beats_baselines_on_synthetic():
    with tempfile.TemporaryDirectory() as tmp:
        csv = write_synthetic(str(Path(tmp) / "s.csv"), n_interactions=25000, seed=2)
        m = run(_tiny_cfg(csv, str(Path(tmp) / "art")), save=True)
        assert m["retrieval"]["recall@20"] > m["retrieval"]["popularity_recall@20"]
        assert (Path(tmp) / "art" / "metrics.json").exists()
