#!/usr/bin/env python
"""Train + evaluate ShortRec on the real dataset (or any configured CSV).

Usage:
    python scripts/run_pipeline.py --config config.yaml

Download KuaiRand-Pure first (https://kuairand.com) and point
`data.interactions_path` in the config at its interaction log.
"""
import argparse
import json

from shortrec.config import load_config
from shortrec.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the ShortRec training pipeline.")
    ap.add_argument("--config", default="config.yaml", help="Path to a YAML config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metrics = run(cfg, save=True)
    print(json.dumps(metrics, indent=2))
    print(f"\nArtifacts + metrics.json written to: {cfg['artifacts_dir']}/")


if __name__ == "__main__":
    main()
