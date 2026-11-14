"""PyTorch models: two-tower retriever and MLP engagement ranker."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    """Two-tower retriever: separate user and item embedding towers.

    User/item vectors are L2-normalized and compared by cosine similarity, so
    the whole catalogue can be scored for a user with one matrix multiply (and
    served from an ANN index over the item vectors). Normalization is important:
    with raw dot products the model can win the sampled-softmax objective just by
    inflating popular items' norms, which collapses to a popularity baseline and
    fails to learn personalization. Cosine + a temperature forces the towers to
    encode *who likes what* rather than *what is popular*.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64,
                 normalize: bool = False, temperature: float = 1.0):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.normalize = normalize
        self.temperature = temperature
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def user_vec(self, u: torch.Tensor) -> torch.Tensor:
        v = self.user_emb(u)
        return F.normalize(v, dim=-1) if self.normalize else v

    def item_vec(self, i: torch.Tensor) -> torch.Tensor:
        v = self.item_emb(i)
        return F.normalize(v, dim=-1) if self.normalize else v

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user_vec(u) * self.item_vec(i)).sum(-1)

    def forward(self, u: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """Sampled-softmax logits over [positive, negatives], scaled by 1/temperature.

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
        return torch.cat([pos_logit, neg_logit], dim=1) / self.temperature

    @torch.no_grad()
    def all_item_vectors(self) -> torch.Tensor:
        w = self.item_emb.weight.detach()
        return F.normalize(w, dim=-1) if self.normalize else w


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
