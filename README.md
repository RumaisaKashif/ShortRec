# ShortRec — Two-Stage Short-Video Recommender

[![CI](https://github.com/RumaisaKashif/ShortRec/actions/workflows/ci.yml/badge.svg)](https://github.com/RumaisaKashif/ShortRec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A compact, production-shaped short-video recommender that mirrors the industrial
two-stage pattern: **candidate generation (retrieval) → ranking**.

**▶ Live demo:** https://huggingface.co/spaces/rumaisakashif/shortrec

```
 user history ──▶ ┌────────────────────┐   top-100    ┌─────────────────────┐   top-10
                  │ Stage 1: Two-Tower │ ───────────▶ │ Stage 2: MLP Ranker │ ─────────▶  recommendations
                  │ retrieval (FAISS   │  candidates  │ predicts long-view  │             (ranked by
                  │ inner-product idx) │              │ (watch-through)     │              P(long-view))
                  └────────────────────┘              └─────────────────────┘
```

- **Stage 1 — Two-tower retrieval.** Separate user/video embedding towers trained
  with a **sampled-softmax loss over exposed negatives** (videos the user was
  shown but didn't engage with — not random unseen items). The objective is the
  same *distinguish-the-true-pair-from-sampled-negatives* idea as **word2vec
  skip-gram-with-negative-sampling**, applied to users↔videos instead of
  words↔contexts. Learned item vectors are indexed in **FAISS** (exact
  inner-product search) for candidate generation. Evaluated with **Recall@K**
  against a popularity baseline.
- **Stage 2 — MLP engagement ranker.** A feed-forward network doing binary
  classification to predict **long-view (watch-through)** from historical
  engagement features. Evaluated with **ROC-AUC** and **GAUC** (grouped, per-user
  AUC) against a logistic-regression baseline, with a feature **ablation**.
- **Evaluation hygiene.** Chronological (time-ordered) train/val/test split, so
  no future interactions leak into the held-out metric; **TEST/serve** features
  use train-only aggregates, and **TRAIN** features use **leave-one-out**
  engagement rates so a row never sees its own label. (Strictly point-in-time
  per-row historical features — aggregates over *preceding* interactions only —
  are a documented refinement; see *Future work*.)

### Methods, precisely (and the NLP caveat)

The ML is **neural collaborative filtering** (two-tower ID embeddings + sampled
softmax) for retrieval and a **feed-forward binary classifier** for ranking —
not NLP. ShortRec models user/video **IDs and engagement signals**, with **no
text, queries, or language models**. The retrieval objective is word2vec-style
(the closest NLP relative), but there is no query understanding or text encoding
here; adding a text/multimodal tower would be the natural extension for a
search-flavoured variant.

## Dataset

Built for **[KuaiRand-Pure](https://kuairand.com)** (Kuaishou short-video feed:
~27K users, ~7.5K videos, ~1.4M interactions, with click / like / follow /
long-view signals). Download it, unzip into `data/`, and point
`data.interactions_path` in `config.yaml` at the interaction log.

No dataset handy? The whole pipeline runs on a built-in **synthetic** generator
with the same schema — see the smoke test below.

## Results (real KuaiRand-Pure)

Trained on the 1,141,112-interaction standard log (26,210 users · 7,538 videos),
chronological split (70/10/20), with leave-one-out train features. These are the
numbers the [live demo](https://huggingface.co/spaces/rumaisakashif/shortrec)
serves (also in `demo/artifacts/metrics.json`):

| Stage | Metric | ShortRec | Baseline |
|---|---|---|---|
| Ranking (long-view) | ROC-AUC | **0.736** | 0.730 (logistic regression) |
| Ranking (long-view) | GAUC | **0.637** | 0.633 (logistic regression) |
| Ranking — ablation | GAUC drop when historical-rate features removed | **−0.080** | — |
| Retrieval | Recall@50 | 0.065 | **0.138 (popularity)** |

Honest reading: the **ranker** is the strong, defensible result — ~0.74 AUC /
0.64 GAUC predicting watch-through, with the ablation showing engagement-history
features contribute most of the GAUC. The **two-tower retriever does not beat the
popularity baseline on Recall** here: short-video engagement is heavily
concentrated on popular items (exposure/feedback-loop bias), so a
personalization-only retriever can't win this particular metric on this dataset.
Reported as-is rather than cherry-picked. Beating it would need a sequential
model (e.g. SASRec/GRU4Rec) or KuaiRand's random-exposure logs for unbiased
training/eval — see *Future work*.

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
get their personalized Top-N with per-item long-view probabilities. It serves the
pretrained model bundled in `demo/artifacts/` (the real KuaiRand model above),
loaded inference-only on CPU; if no bundled model is found it falls back to
training a quick synthetic one so it always boots. Deploy **free** on **Hugging
Face Spaces** — two ways:

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

## Future work

- **Beat the popularity baseline on retrieval** with a sequential recommender
  (SASRec / GRU4Rec) that models watch order, and/or train and evaluate on
  KuaiRand's **random-exposure logs** to remove popularity/feedback-loop bias.
- **A search-flavoured variant**: add a text/multimodal tower (video titles,
  CLIP frame embeddings) for query→video retrieval — the NLP/CV extension this
  ID-only recommender deliberately omits.

## License

MIT — see [LICENSE](LICENSE).
