"""Gradio demo for ShortRec — a free, deployable user-facing interface.

Run locally:        python demo/app.py
Deploy (free):      Hugging Face Spaces (see demo/README_SPACE.md).

On startup it loads trained artifacts from $SHORTREC_ARTIFACTS (or ./artifacts).
If none exist, it trains a small model on synthetic data so the Space always
boots with a working demo — replace artifacts with a real run for real results.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

# Make the `shortrec` package importable when run from a cloned repo / Space.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shortrec.config import load_config  # noqa: E402
from shortrec.pipeline import run  # noqa: E402
from shortrec.recommender import Recommender  # noqa: E402
from shortrec.synthetic import write_synthetic  # noqa: E402

# Hugging Face ZeroGPU: importing `spaces` marks the Space as ZeroGPU-ready, and
# ZeroGPU expects at least one @spaces.GPU-registered function. Our recommender
# inference is CPU-light and runs in-process on CPU, so this warmup is only here
# to satisfy ZeroGPU's registration; it is not used in the normal request path.
# Guarded so the app still runs locally (where `spaces` isn't installed).
try:
    import spaces  # noqa: E402

    @spaces.GPU
    def _gpu_warmup():
        import torch

        return float(torch.zeros(1).sum())
except ImportError:
    pass


HERE = Path(__file__).resolve().parent


def _ensure_artifacts() -> str:
    """Return a dir with trained artifacts.

    Prefers the model bundled in the repo (demo/artifacts) so the Space loads and
    serves without training at boot — important on ZeroGPU, where training would
    trigger a CUDA init outside a @spaces.GPU context. Only trains (locally) if
    no bundled or configured artifacts exist.
    """
    env = os.environ.get("SHORTREC_ARTIFACTS")
    for cand in [env, str(HERE / "artifacts"), "artifacts", str(HERE.parent / "artifacts")]:
        if cand and (Path(cand) / "meta.json").exists():
            return cand

    print("No artifacts found — training a quick model on synthetic data...")
    tmp = tempfile.mkdtemp(prefix="shortrec_")
    csv = write_synthetic(str(Path(tmp) / "synth.csv"), seed=0)
    cfg = load_config()
    cfg["data"]["interactions_path"] = csv
    cfg["artifacts_dir"] = str(Path(tmp) / "artifacts")
    cfg["retriever"].update(emb_dim=32, epochs=3)
    cfg["ranker"].update(epochs=3)
    run(cfg, save=True)
    return cfg["artifacts_dir"]


# Force CPU: the demo is inference-only and CPU-light. On ZeroGPU this also
# avoids any CUDA probing (torch.cuda.is_available / optimizer accelerator
# checks) that its PyTorch patch would reject outside a @spaces.GPU context.
DEVICE = torch.device("cpu")
ARTIFACTS = _ensure_artifacts()
REC = Recommender.from_dir(ARTIFACTS, DEVICE)
USER_CHOICES = [str(u) for u in REC.known_users[:500]]
METRICS = REC.meta.get("metrics", {})


def recommend(user_id: str, k: int, n_candidates: int) -> pd.DataFrame:
    recs = REC.recommend(int(user_id), k=int(k), n_candidates=int(n_candidates))
    rows = [
        {
            "rank": i + 1,
            "video_id": r["item_id"],
            "P(long-view)": round(r["p_longview"], 4),
            "retriever_score": None if r["retriever_score"] is None else round(r["retriever_score"], 3),
        }
        for i, r in enumerate(recs)
    ]
    return pd.DataFrame(rows)


def _metrics_md() -> str:
    if not METRICS:
        return "_No metrics found in artifacts._"
    ret, rank = METRICS.get("retrieval", {}), METRICS.get("ranking", {})
    return (
        f"**Dataset:** {METRICS.get('n_users','?')} users · {METRICS.get('n_items','?')} videos · "
        f"{METRICS.get('n_interactions','?')} interactions  \n"
        f"**Retrieval:** Recall@50 `{ret.get('recall@50')}` vs popularity "
        f"`{ret.get('popularity_recall@50')}` (lift `{ret.get('lift_vs_popularity')}x`)  \n"
        f"**Ranking:** MLP GAUC `{rank.get('mlp_gauc')}` vs logistic-regression `{rank.get('logreg_gauc')}`"
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ShortRec") as demo:
        gr.Markdown(
            "# 🎬 ShortRec — Two-Stage Short-Video Recommender\n"
            "Two-tower retrieval → MLP engagement ranking. Pick a user to see their "
            "personalized Top-N (ranked by predicted long-view probability)."
        )
        gr.Markdown(_metrics_md())
        with gr.Row():
            user = gr.Dropdown(USER_CHOICES, value=USER_CHOICES[0], label="User ID", scale=2)
            k = gr.Slider(1, 20, value=10, step=1, label="Top-N")
            ncand = gr.Slider(20, 500, value=100, step=10, label="Retrieval candidates")
        btn = gr.Button("Recommend", variant="primary")
        out = gr.Dataframe(label="Recommendations", interactive=False)
        btn.click(recommend, inputs=[user, k, ncand], outputs=out)
        demo.load(recommend, inputs=[user, k, ncand], outputs=out)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
