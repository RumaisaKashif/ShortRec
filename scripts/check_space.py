#!/usr/bin/env python
"""Poll a Hugging Face Space's runtime stage after a deploy (for CI monitoring).

Exits 0 once the Space is RUNNING, non-zero on a build/runtime error or timeout,
so the GitHub Actions log reports the live deploy outcome without anyone having
to open the Space. Requires HF_TOKEN + HF_USERNAME in the environment.
"""
from __future__ import annotations

import os
import sys
import time

OK = {"RUNNING", "SLEEPING"}
ERR = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}
# Stages that prove our just-uploaded commit started rebuilding. We only trust a
# terminal stage after seeing one of these, so we don't read the PREVIOUS
# container's stale RUNNING/ERROR before the rebuild kicks in.
PROGRESS = {"BUILDING", "APP_STARTING", "RUNNING_BUILDING"}


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    user = os.environ.get("HF_USERNAME")
    space = os.environ.get("HF_SPACE", "shortrec")
    if not token or not user:
        raise SystemExit("HF_TOKEN and HF_USERNAME are required.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = f"{user}/{space}"
    timeout = int(os.environ.get("TIMEOUT", "720"))

    deadline = time.time() + timeout
    last = None
    seen_progress = False
    while time.time() < deadline:
        stage = api.get_space_runtime(repo_id).stage
        if stage != last:
            print(f"stage: {stage}", flush=True)
            last = stage
        if stage in PROGRESS:
            seen_progress = True
        elif seen_progress and stage in OK:
            print(f"Space is up: https://huggingface.co/spaces/{repo_id}")
            return 0
        elif seen_progress and stage in ERR:
            print(f"::error::Space entered {stage}. See logs at "
                  f"https://huggingface.co/spaces/{repo_id}?logs=container")
            return 1
        # Before the rebuild starts, any terminal stage is the previous
        # container's — keep waiting for BUILDING/APP_STARTING.
        time.sleep(10)

    print("::error::Timed out waiting for the rebuild to finish.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
