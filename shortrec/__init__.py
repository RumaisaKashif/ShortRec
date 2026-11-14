"""ShortRec: a two-stage short-video recommender (retrieval -> ranking).

Stage 1: a two-tower retriever learns user/video embeddings and generates
         top-K candidates (evaluated with Recall@K vs a popularity baseline).
Stage 2: an MLP engagement ranker re-ranks candidates to predict long-view
         (watch-through), evaluated with GAUC vs a logistic-regression baseline.

Designed for KuaiRand-Pure, with leakage-safe temporal splits and negatives
sampled from logged impressions (shown-but-not-engaged), not random unseen items.
"""

__version__ = "0.1.0"
