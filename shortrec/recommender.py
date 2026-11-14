"""Inference wrapper: turn a loaded bundle into live top-N recommendations.

Reusable by the Gradio demo, a batch job, or a REST endpoint.
"""
from __future__ import annotations

import numpy as np
import torch

from .artifacts import load_bundle
from .features import build_features


class Recommender:
    def __init__(self, bundle: dict, device):
        self.device = device
        self.tt = bundle["two_tower"]
        self.ranker = bundle["ranker"]
        self.mean = bundle["scaler_mean"]
        self.scale = bundle["scaler_scale"]
        self.id_maps = bundle["id_maps"]
        self.stats = bundle["stats"]
        self.item_duration = bundle["item_duration"]
        self.meta = bundle["meta"]
        self._idx2item = {i: v for v, i in self.id_maps.item2idx.items()}
        self._item_mat = self.tt.all_item_vectors().to(device)

    @classmethod
    def from_dir(cls, artifacts_dir: str, device):
        return cls(load_bundle(artifacts_dir, device), device)

    @property
    def known_users(self) -> list[int]:
        return list(self.id_maps.user2idx)

    @torch.no_grad()
    def recommend(self, user_id: int, k: int = 10, n_candidates: int = 100) -> list[dict]:
        """Two-stage recommend: retrieve n_candidates, then rank, return top-k."""
        if user_id not in self.id_maps.user2idx:
            return self._popular(k)

        uidx = self.id_maps.user2idx[user_id]
        uv = self.tt.user_vec(torch.tensor([uidx], device=self.device)).squeeze(0)
        scores = self._item_mat @ uv                      # (n_items,)
        n_candidates = min(n_candidates, scores.numel())
        top = torch.topk(scores, n_candidates)
        cand_idx = top.indices.cpu().numpy()
        cand_scores = top.values.cpu().numpy()
        cand_items = np.array([self._idx2item[i] for i in cand_idx])

        feats = build_features(
            np.full(n_candidates, user_id), cand_items,
            self.stats, self.item_duration,
        )
        p = self._rank(feats)
        order = np.argsort(-p)[:k]
        return [
            {
                "item_id": int(cand_items[j]),
                "p_longview": float(p[j]),
                "retriever_score": float(cand_scores[j]),
            }
            for j in order
        ]

    def _rank(self, feats: np.ndarray) -> np.ndarray:
        Xs = ((feats - self.mean) / self.scale).astype(np.float32)
        logits = self.ranker(torch.from_numpy(Xs).to(self.device))
        return torch.sigmoid(logits).cpu().numpy()

    def _popular(self, k: int) -> list[dict]:
        """Cold-user fallback: rank catalogue by historical engagement rate."""
        items = sorted(
            self.stats.item_rate,
            key=lambda it: self.stats.item_rate[it] * np.log1p(self.stats.item_count.get(it, 0)),
            reverse=True,
        )[:k]
        return [
            {"item_id": int(it), "p_longview": float(self.stats.item_rate[it]), "retriever_score": None}
            for it in items
        ]
