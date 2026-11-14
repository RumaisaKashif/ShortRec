"""Stage 1: two-tower retriever training, scoring, and the popularity baseline."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import IdMaps
from .metrics import recall_at_k
from .models import TwoTower


def build_faiss_index(item_vectors: np.ndarray):
    """Build a FAISS inner-product index over item vectors (None if unavailable).

    The catalogue is small (thousands of videos), so IndexFlatIP (exact
    inner-product search) is the right choice; the same interface scales to an
    approximate index (IVF/HNSW) for a production-sized catalogue.
    """
    try:
        import faiss
    except ImportError:
        return None
    mat = np.ascontiguousarray(item_vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index


def _sample_negatives(
    user_orig: np.ndarray,
    exposed_neg: dict[int, np.ndarray],
    item2idx: dict[int, int],
    all_item_idx: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample k negative item *indices* per positive from exposed impressions.

    Falls back to a random catalogue item when a user has too few exposed
    negatives, so every positive gets k negatives.
    """
    out = np.empty((len(user_orig), k), dtype=np.int64)
    for row, u in enumerate(user_orig):
        pool = exposed_neg.get(int(u))
        if pool is not None and len(pool) > 0:
            chosen = pool[rng.integers(0, len(pool), size=k)]
            out[row] = [item2idx[int(c)] for c in chosen]
        else:
            out[row] = all_item_idx[rng.integers(0, len(all_item_idx), size=k)]
    return out


def train_two_tower(
    pos_df,
    exposed_neg: dict[int, np.ndarray],
    id_maps: IdMaps,
    cfg: dict,
    device,
    rng: np.random.Generator,
) -> TwoTower:
    """Train the retriever with a sampled-softmax loss over exposed negatives."""
    model = TwoTower(id_maps.n_users, id_maps.n_items, cfg["emb_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    u_idx = torch.as_tensor(id_maps.map_users(pos_df["user_id"]), dtype=torch.long)
    p_idx = torch.as_tensor(id_maps.map_items(pos_df["item_id"]), dtype=torch.long)
    user_orig = pos_df["user_id"].to_numpy()
    all_item_idx = np.arange(id_maps.n_items)
    loader = DataLoader(
        TensorDataset(torch.arange(len(pos_df)), u_idx, p_idx),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    model.train()
    for _ in range(cfg["epochs"]):
        for rows, ub, pb in loader:
            neg = _sample_negatives(
                user_orig[rows.numpy()], exposed_neg, id_maps.item2idx,
                all_item_idx, cfg["num_negatives"], rng,
            )
            ub, pb = ub.to(device), pb.to(device)
            nb = torch.as_tensor(neg, dtype=torch.long, device=device)
            logits = model(ub, pb, nb)
            target = torch.zeros(len(ub), dtype=torch.long, device=device)  # pos at col 0
            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def retrieve_topk(
    model: TwoTower,
    user_orig_ids: list[int],
    id_maps: IdMaps,
    seen: dict[int, set],
    k: int,
    device,
) -> dict[int, list[int]]:
    """Return user_id -> top-k item ids (original ids), excluding train-seen items.

    Candidate generation runs through a FAISS inner-product index when available
    (over-fetching k + |seen| so seen items can be dropped), falling back to a
    torch top-k otherwise.
    """
    item_mat = model.all_item_vectors().to(device)          # (n_items, d)
    idx2item = {i: v for v, i in id_maps.item2idx.items()}
    n_items = item_mat.shape[0]
    index = build_faiss_index(item_mat.detach().cpu().numpy())
    out: dict[int, list[int]] = {}
    for u in user_orig_ids:
        if u not in id_maps.user2idx:
            continue
        uv = model.user_vec(torch.tensor([id_maps.user2idx[u]], device=device)).squeeze(0)
        seen_idx = {id_maps.item2idx[it] for it in seen.get(u, ()) if it in id_maps.item2idx}
        if index is not None:
            need = min(n_items, k + len(seen_idx))
            q = uv.detach().cpu().numpy().astype(np.float32)[None, :]
            found = index.search(q, need)[1][0]
            cand = [i for i in found.tolist() if i not in seen_idx][:k]
        else:
            scores = item_mat @ uv
            for i in seen_idx:
                scores[i] = float("-inf")
            cand = torch.topk(scores, min(k, scores.numel())).indices.tolist()
        out[u] = [idx2item[i] for i in cand]
    return out


def evaluate_retrieval(model, test_pos, id_maps, seen, k, device) -> float:
    """Mean Recall@k on held-out positives."""
    relevant: dict[int, set] = {}
    for u, it in zip(test_pos["user_id"], test_pos["item_id"], strict=True):
        relevant.setdefault(int(u), set()).add(int(it))
    ranked = retrieve_topk(model, list(relevant), id_maps, seen, k, device)
    return recall_at_k(ranked, relevant)


def popularity_recall(train_df, test_pos, seen, k) -> float:
    """Popularity baseline: recommend the globally most-engaged items."""
    pop = (
        train_df[train_df["engage"] == 1]["item_id"].value_counts().index.tolist()
    )
    relevant: dict[int, set] = {}
    for u, it in zip(test_pos["user_id"], test_pos["item_id"], strict=True):
        relevant.setdefault(int(u), set()).add(int(it))
    ranked: dict[int, list[int]] = {}
    for u in relevant:
        s = seen.get(u, set())
        recs, i = [], 0
        while len(recs) < k and i < len(pop):
            if pop[i] not in s:
                recs.append(pop[i])
            i += 1
        ranked[u] = recs
    return recall_at_k(ranked, relevant)
