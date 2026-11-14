"""PyTorch models: two-tower retriever and MLP engagement ranker."""
from __future__ import annotations

import torch
import torch.nn as nn


class TwoTower(nn.Module):
    """Two-tower retriever: separate user and item embedding towers.

    Relevance is the dot product of the user and item vectors, so the whole
    item catalogue can be scored for a user with one matrix multiply (and, at
    larger scale, served from an ANN index built over the item vectors).
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def user_vec(self, u: torch.Tensor) -> torch.Tensor:
        return self.user_emb(u)

    def item_vec(self, i: torch.Tensor) -> torch.Tensor:
        return self.item_emb(i)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user_vec(u) * self.item_vec(i)).sum(-1)

    def forward(self, u: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """Sampled-softmax logits over [positive, negatives].

        u:   (B,)        user indices
        pos: (B,)        positive item indices
        neg: (B, K)      sampled exposed-negative item indices
        returns (B, 1+K) logits; the positive is always column 0.
        """
        uv = self.user_vec(u)                       # (B, d)
        pv = self.item_vec(pos)                      # (B, d)
        nv = self.item_vec(neg)                      # (B, K, d)
        pos_logit = (uv * pv).sum(-1, keepdim=True)  # (B, 1)
        neg_logit = torch.bmm(nv, uv.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return torch.cat([pos_logit, neg_logit], dim=1)

    @torch.no_grad()
    def all_item_vectors(self) -> torch.Tensor:
        return self.item_emb.weight.detach()


class MLPRanker(nn.Module):
    """Small MLP that predicts long-view (watch-through) from dense features."""

    def __init__(self, in_dim: int, hidden_dims=(128, 64), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits
