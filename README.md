# ShortRec — Two-Stage Short-Video Recommender

[![CI](https://github.com/RumaisaKashif/ShortRec/actions/workflows/ci.yml/badge.svg)](https://github.com/RumaisaKashif/ShortRec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A compact, production-shaped short-video recommender that mirrors the industrial
two-stage pattern: **candidate generation (retrieval) → ranking**.

```
 user history ──▶ ┌────────────────────┐   top-100    ┌─────────────────────┐   top-10
                  │ Stage 1: Two-Tower │ ───────────▶ │ Stage 2: MLP Ranker │ ─────────▶  recommendations
                  │ retrieval (FAISS-  │  candidates  │ predicts long-view  │             (ranked by
                  │ ready embeddings)  │              │ (watch-through)     │              P(long-view))
                  └────────────────────┘              └─────────────────────┘
```

- **Stage 1 — Two-tower retrieval.** Separate user/video embedding towers trained
  with a sampled-softmax loss over **exposed negatives** (videos the user was
  shown but didn't engage with — not random unseen items). Evaluated with
  **Recall@50** against a popularity baseline.
- **Stage 2 — MLP engagement ranker.** Re-ranks candidates to predict
  **long-view (watch-through)** from the retrieval score plus train-only
  historical engagement features. Evaluated with **GAUC** (grouped, per-user AUC)
  against a logistic-regression baseline, with a feature **ablation**.
- **Leakage-safe by construction.** Chronological train/val/test split; all
  history features computed from the training period only.

> **Note on metrics.** The numbers the pipeline reports are *your measured
> results* on whatever data you run. Run it, then quote the real figures — don't
> ship unmeasured numbers.

## Dataset

Built for **[KuaiRand-Pure](https://kuairand.com)** (Kuaishou short-video feed:
~27K users, ~7.5K videos, ~1.4M interactions, with click / like / follow /
long-view signals). Download it, unzip into `data/`, and point
`data.interactions_path` in `config.yaml` at the interaction log.

No dataset handy? The whole pipeline runs on a built-in **synthetic** generator
with the same schema — see the smoke test below.

## Quickstart

```bash
# 1. Install (CPU torch)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install -e ".[dev,demo]"

# 2. Verify everything runs end-to-end on synthetic data (no download needed)
python scripts/smoke_test.py

# 3. Train + evaluate on the real dataset
python scripts/run_pipeline.py --config config.yaml   # writes artifacts/ + metrics.json

# 4. Launch the demo locally (http://localhost:7860)
python demo/app.py
```

## Demo (free, deployable)

A **Gradio** app (`demo/app.py`) exposes the two-stage recommender: pick a user,
get their personalized Top-N with per-item long-view probabilities. It loads
`artifacts/` if present, otherwise trains a quick synthetic model so it always
boots. Deploy **free** on **Hugging Face Spaces** — two ways:

**Automated (CI/CD).** The `Deploy to Hugging Face Spaces` workflow deploys after
CI passes on `main` (and on manual dispatch). One-time setup in
**Settings → Secrets and variables → Actions**:
- add a **secret** `HF_TOKEN` — a Hugging Face *write* token
  ([create one](https://huggingface.co/settings/tokens)), and
- add a **variable** `HF_USERNAME` — your Hugging Face username.

The workflow creates the Space and pushes to it; your token stays encrypted in
GitHub and is never exposed.

**One-liner (local).** Deploy from your machine without touching GitHub secrets:
```bash
pip install huggingface_hub
HF_TOKEN=hf_xxx python scripts/deploy_space.py --username <your-hf-username>
```

Space config lives in [`demo/README_SPACE.md`](demo/README_SPACE.md).

## Project layout

```
shortrec/
  config.py        # defaults + YAML loader
  data.py          # load, temporal split, history stats, exposed-negative sampling
  models.py        # TwoTower + MLPRanker (PyTorch)
  features.py      # shared ranker features (train/serve parity)
  retrieval.py     # Stage 1: train / score / popularity baseline
  ranking.py       # Stage 2: train / eval / logistic-regression baseline
  pipeline.py      # end-to-end orchestration -> metrics.json + artifacts
  recommender.py   # inference wrapper (used by the demo / any server)
  artifacts.py     # save / load trained bundle
scripts/           # run_pipeline.py (real data), smoke_test.py (synthetic)
demo/              # Gradio app + Hugging Face Space config
tests/             # unit + fast integration tests
.github/workflows/ # CI: ruff + pytest + smoke test
```

## Development

```bash
ruff check .      # lint
pytest            # unit + integration tests
```

CI (GitHub Actions) runs lint, tests, and the synthetic end-to-end smoke test on
every push and pull request.

## License

MIT — see [LICENSE](LICENSE).
