# Course website

This folder contains a small Jupyter Book site that can be published via GitHub Pages.

Notebook source policy:

- Ground-truth notebooks live in `../notebooks/sessions/`.
- `website/content/*.ipynb` is generated from ground-truth notebooks by:
  - `python scripts/sync_notebooks_to_website.py`
- The GitHub Actions deployment workflow runs this sync step on every push before building the book.
- Sync behavior:
  - prefers `session_x_web.ipynb` when present
  - otherwise uses `session_x.ipynb`
  - requires a paired `session_x_empty.ipynb`
  - rewrites the Colab button link in published notebook copies to `session_x_empty.ipynb`

## Build locally

From the repo root:

```bash
pip install -r website/requirements.txt
python scripts/sync_notebooks_to_website.py
jupyter-book build website
```

The generated HTML will be in:

- `website/_build/html/`
