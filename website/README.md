# Course website

This folder contains a small Jupyter Book site that can be published via GitHub Pages.

Notebook source policy:

- Ground-truth notebooks live in `../notebooks/sessions/`.
- `website/content/` contains symlinks to those notebooks so there is no duplicated notebook content.

## Build locally

From the repo root:

```bash
pip install -r website/requirements.txt
jupyter-book build website
```

The generated HTML will be in:

- `website/_build/html/`
