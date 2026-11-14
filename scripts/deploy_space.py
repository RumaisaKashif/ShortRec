#!/usr/bin/env python
"""Deploy the ShortRec Gradio demo to a (free) Hugging Face Space.

Reusable from a laptop or from CI. Requires an HF **write** token in the
HF_TOKEN environment variable (never hard-code it).

    pip install huggingface_hub
    HF_TOKEN=hf_xxx python scripts/deploy_space.py --username <your-hf-username>

It creates the Space (Gradio SDK) if needed and uploads the repo, using
`demo/README_SPACE.md` as the Space's README (front-matter sets
`app_file: demo/app.py`) and `demo/requirements.txt` as the Space requirements.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", "*.pyc", "*.egg-info", ".pytest_cache", ".ruff_cache",
    "data", "artifacts", ".venv", "venv",
)


def _stage() -> str:
    """Copy the repo into a temp dir and set the Space README + requirements."""
    tmp = tempfile.mkdtemp(prefix="hf_space_")
    dst = Path(tmp) / "space"
    shutil.copytree(REPO_ROOT, dst, ignore=IGNORE)
    shutil.copy(dst / "demo" / "README_SPACE.md", dst / "README.md")       # front-matter
    shutil.copy(dst / "demo" / "requirements.txt", dst / "requirements.txt")  # gradio + deps
    return str(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy the ShortRec demo to Hugging Face Spaces.")
    ap.add_argument("--username", required=True, help="Your Hugging Face username.")
    ap.add_argument("--space", default="shortrec", help="Space name (default: shortrec).")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (a Hugging Face *write* token) in the environment.")

    from huggingface_hub import HfApi

    repo_id = f"{args.username}/{args.space}"
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    folder = _stage()
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="space",
                      commit_message="Deploy ShortRec demo")
    print(f"Deployed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
