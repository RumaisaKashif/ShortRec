#!/usr/bin/env python
"""End-to-end smoke test on synthetic data — no external dataset required.

Generates a KuaiRand-shaped synthetic log, runs the full
retrieve -> rank -> evaluate -> save pipeline on a tiny config, and asserts the
learned models beat their baselines (sanity check, not a benchmark). Also used
by CI so the whole pipeline is exercised on every push.
"""
import sys
import tempfile
from pathlib import Path

from shortrec.config import load_config
from shortrec.pipeline import run
from shortrec.recommender import Recommender
from shortrec.synthetic import write_synthetic
from shortrec.utils import get_device


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        csv = write_synthetic(str(Path(tmp) / "synth.csv"), seed=0)
        cfg = load_config()
        cfg["data"]["interactions_path"] = csv
        cfg["artifacts_dir"] = str(Path(tmp) / "artifacts")
        # Small + fast settings for CI (enough gradient steps on little data).
        cfg["retriever"].update(emb_dim=32, epochs=15, batch_size=256, lr=5e-3,
                                num_negatives=8, recall_k=50)
        cfg["ranker"].update(epochs=5, batch_size=256)

        metrics = run(cfg, save=True)
        print("metrics:", metrics)

        rec = metrics["retrieval"]
        assert rec["recall@50"] > rec["popularity_recall@50"], "retriever did not beat popularity"
        assert metrics["ranking"]["mlp_gauc"] and metrics["ranking"]["mlp_gauc"] > 0.5, "ranker GAUC <= 0.5"

        # Inference wrapper loads artifacts and returns ranked recommendations.
        r = Recommender.from_dir(cfg["artifacts_dir"], get_device())
        recs = r.recommend(r.known_users[0], k=10)
        assert len(recs) == 10 and all("p_longview" in x for x in recs), "bad recommendations"

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
