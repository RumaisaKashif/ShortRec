"""Stage 2: MLP engagement ranker + logistic-regression baseline.

Operates on prebuilt feature matrices (see `features.build_features`), so this
module stays independent of how features are produced.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .metrics import auc, gauc
from .models import MLPRanker


def train_ranker(X: np.ndarray, y: np.ndarray, cfg: dict, device):
    """Fit a StandardScaler + MLP ranker. Returns (model, scaler)."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    model = MLPRanker(X.shape[1], tuple(cfg["hidden_dims"]), cfg.get("dropout", 0.1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y.astype(np.float32))),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    model.train()
    for _ in range(cfg["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, scaler


@torch.no_grad()
def predict(model, scaler, X: np.ndarray, device) -> np.ndarray:
    Xs = scaler.transform(X).astype(np.float32)
    logits = model(torch.from_numpy(Xs).to(device))
    return torch.sigmoid(logits).cpu().numpy()


def evaluate_ranker(model, scaler, X, y, user_ids, device) -> dict:
    scores = predict(model, scaler, X, device)
    return {"auc": auc(y, scores), "gauc": gauc(user_ids, y, scores)}


def logreg_baseline(X_train, y_train, X_test, y_test, user_ids_test) -> dict:
    """Logistic-regression ranker on the same features (scaled)."""
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(scaler.transform(X_train), y_train)
    scores = clf.predict_proba(scaler.transform(X_test))[:, 1]
    return {"auc": auc(y_test, scores), "gauc": gauc(user_ids_test, y_test, scores)}
