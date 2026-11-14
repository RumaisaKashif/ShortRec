---
title: ShortRec Demo
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: demo/app.py
pinned: false
---

# ShortRec Demo (Hugging Face Space)

This is the Space configuration for the ShortRec demo. To deploy **for free**:

1. Create a new Space at https://huggingface.co/new-space → SDK: **Gradio**.
2. Push this repository to the Space (or link the GitHub repo):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/shortrec
   git push space main
   ```
3. In the Space, rename/replace the Space's `README.md` with **this file's
   front-matter** (the block above), so it uses `app_file: demo/app.py` and the
   Gradio SDK, and point its requirements at `demo/requirements.txt`
   (or copy those lines into the Space's root `requirements.txt`).

On first boot, with no trained artifacts committed, the app trains a small model
on synthetic data so the demo works immediately. For real numbers, run the
pipeline locally and commit the `artifacts/` folder (or upload it to the Space).
