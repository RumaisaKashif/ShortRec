"""Synthetic KuaiRand-like interaction log.

Generates data with the SAME schema as KuaiRand-Pure so the real data pipeline
(`data.load_interactions`) runs on it unchanged. Used by the smoke test to
verify the full retrieve -> rank -> eval pipeline end-to-end without the real
dataset (which must be downloaded separately from https://kuairand.com).

The generator plants genuine structure — latent user/item factors drive
engagement — so a trained model should beat the popularity / logistic-regression
baselines on synthetic data too (a sanity check, not a benchmark).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_kuairand(
    n_users: int = 800,
    n_items: int = 500,
    n_interactions: int = 40_000,
    latent_dim: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame matching KuaiRand-Pure's key columns."""
    rng = np.random.default_rng(seed)

    user_factors = rng.normal(size=(n_users, latent_dim))
    item_factors = rng.normal(size=(n_items, latent_dim))
    item_pop = rng.gamma(2.0, 1.0, size=n_items)
    item_pop /= item_pop.sum()
    item_bias = rng.normal(scale=0.2, size=n_items)   # mild per-item quality, NOT tied to popularity
    item_duration = rng.integers(5_000, 60_000, size=n_items)  # ms

    # Impressions: a light popularity skew blended with uniform exposure. The
    # popularity component keeps the popularity baseline non-trivial, while
    # engagement itself is driven mainly by personalization (below), so a
    # two-tower that learns user<->item affinity should beat popularity.
    users = rng.integers(0, n_users, size=n_interactions)
    expose_p = 0.85 * np.full(n_items, 1.0 / n_items) + 0.15 * item_pop
    expose_p /= expose_p.sum()
    items = rng.choice(n_items, size=n_interactions, p=expose_p)

    # Strong, learnable affinity signal (this is a sanity fixture, not a
    # benchmark): engagement depends mostly on user<->item latent similarity.
    affinity = np.einsum("ij,ij->i", user_factors[users], item_factors[items])
    affinity /= np.sqrt(latent_dim)   # ~ unit variance
    logit = 4.0 * affinity + item_bias[items]
    p_click = 1.0 / (1.0 + np.exp(-(logit + 0.3)))
    click = (rng.random(n_interactions) < p_click).astype(np.int8)
    # long_view only possible if clicked; a harder (higher) affinity bar.
    p_long = 1.0 / (1.0 + np.exp(-(logit - 0.8)))
    long_view = (click & (rng.random(n_interactions) < p_long)).astype(np.int8)

    # Monotonically increasing timestamps (ms) so temporal split is meaningful.
    base = 1_700_000_000_000
    ts = np.sort(rng.integers(0, 14 * 24 * 3600 * 1000, size=n_interactions)) + base

    return pd.DataFrame(
        {
            "user_id": users.astype(np.int64),
            "video_id": items.astype(np.int64),
            "time_ms": ts.astype(np.int64),
            "is_click": click,
            "long_view": long_view,
            "duration_ms": item_duration[items].astype(np.int64),
        }
    )


def write_synthetic(path: str, **kwargs) -> str:
    """Generate and save a synthetic log to `path` (CSV). Returns the path."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    make_synthetic_kuairand(**kwargs).to_csv(path, index=False)
    return path
