# Continuation notes (RL course materials)

This file summarizes what was set up in this workspace and what to do next to continue/publish it (e.g., on GitHub Pages), without needing the full chat history.

Date: 2026-02-10

---

## What was added/changed

### 1) MDP / Dynamic Programming solution notebook

File:
- `notebooks/sessions/session_02_mdp_dynamic_programming/session_02_mdp_dynamic_programming_blank.ipynb`

Content added:
- Tabular MDP representation (`TabularMDP`) with explicit transition lists.
- DP algorithms:
  - Iterative policy evaluation
  - Policy iteration
  - Value iteration
  - Diagnostics variants returning per-sweep deltas
  - Modified policy iteration
- Practice environments/tasks:
  - 4x4 Gridworld (deterministic)
  - Slippery Gridworld (action slip dynamics)
  - Gambler’s Problem (value iteration)
  - Jack’s Car Rental (scaled-down, Poisson-based policy iteration; “car moving” problem)
- Visualization helpers:
  - Printed value/policy grids
  - Heatmaps for values
  - Convergence plots (delta per sweep)
- Extra “Suggested exercises” and “More tasks” sections.

Notes:
- Notebook execution was not fully validated end-to-end because notebook kernel execution was cancelled at least once. If you want to validate, open the notebook and “Run all” in a Python kernel with `numpy`/`matplotlib` installed.

---

### 2) Docker environment (local Jupyter)

File:
- `Dockerfile`

Behavior:
- Builds an image with Python 3.11 and installs dependencies from `requirements.txt`.
- Runs JupyterLab on port 8888 with **token/password disabled** (intended for local use).

Important:
- Running Jupyter without a token is unsafe on untrusted networks. Use it only on localhost or behind a firewall.

Run commands:
- Build: `docker build -t drl-course .`
- Run: `docker run --rm -p 8888:8888 drl-course`

Related:
- `.dockerignore` added to speed up builds.

---

### 3) Python dependencies for students / Colab

File:
- `requirements.txt`

Contains:
- Jupyter stack: `jupyterlab`, `ipykernel`
- Scientific: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`
- RL: `torch` (CPU via pip), `gymnasium[classic-control,toy-text]`, `pygame`

Colab usage hint:
- In Colab, students can run `!pip install -r requirements.txt` (Colab usually has torch already, but this keeps installs consistent).

---

### 4) GitHub Pages website (Jupyter Book)

Goal:
- Create a UVA-DLC-notebooks-like static webpage from notebooks, hosted on GitHub Pages.

Folder:
- `website/`

Key files:
- `website/_config.yml` (Jupyter Book config; notebooks execution disabled for faster builds)
- `website/_toc.yml` (table of contents)
- `website/intro.md` (landing page)
- `website/content/session_01_k_armed_bandit_blank.ipynb` (copied from `notebooks/sessions/session_01_k_armed_bandit/session_01_k_armed_bandit_blank.ipynb`)
- `website/assets/logo.png` (copied from `notebooks/shared_assets/logo.png`)
- `website/requirements.txt` (site build requirements)

Important build note (Node.js):
- Jupyter Book 2.x requires Node.js.
- To avoid forcing Node.js for local builds, `website/requirements.txt` pins **Jupyter Book 1.x**:
  - `jupyter-book>=1.0.0,<2`

Local build:
- `pip install -r website/requirements.txt`
- `jupyter-book build website`
- Open: `website/_build/html/index.html`

Gitignore:
- `.gitignore` was created and includes `website/_build/` and cache folders.

---

### 5) GitHub Actions workflow for Pages

File:
- `.github/workflows/deploy-website.yml`

What it does:
- On push to `main` (and manual trigger), installs Python, installs `website/requirements.txt`, builds the book, uploads artifact, deploys to GitHub Pages.

IMPORTANT:
- When checked locally, this folder was **not a git repo** yet.
  - You must put it in a GitHub repository for Actions/Pages to work.
- The workflow triggers on branch `main`. If your repo uses `master`, update the workflow.

---

## Current status / caveats

1) The website builds locally with Jupyter Book 1.x.
2) Publishing to GitHub Pages is not active until:
   - This project is in a GitHub repo
   - GitHub Pages is enabled in repo settings
   - Branch name matches workflow trigger (`main` by default)
3) The MDP solution notebook has a lot of new content; run-all validation is still recommended.

---

## Next steps (recommended)

### A) Put the project into GitHub

Option 1: Create a new repo and push
- `git init`
- `git add .`
- `git commit -m "Initial course materials"`
- Create a GitHub repo and add remote
- `git push -u origin main`

Option 2: If you already have a repo
- Copy/merge:
  - `website/`
  - `.github/workflows/deploy-website.yml`
  - `.gitignore`

### B) Enable GitHub Pages (once repo exists)

- GitHub repo settings → Pages
- Source: GitHub Actions (recommended)

Then pushing to `main` should publish.

### C) Improve the “UVA-like” feel (optional)

Ideas:
- Add badges/links on `website/intro.md`:
  - “View on GitHub”
  - “Open in Colab” (link to notebook in repo)
- Rename sidebar entry by using a markdown wrapper page with a nicer title, or configure `mystnb`/titles.

### D) Add more practicals to the site

- Copy more notebooks into `website/content/`
- Add them to `website/_toc.yml`

---

## Quick file map

- Practical notebooks:
  - `notebooks/sessions/session_01_k_armed_bandit/session_01_k_armed_bandit_blank.ipynb`
  - `notebooks/sessions/session_02_mdp_dynamic_programming/session_02_mdp_dynamic_programming_blank.ipynb`

- Docker + deps:
  - `Dockerfile`
  - `requirements.txt`
  - `.dockerignore`

- Website:
  - `website/_config.yml`
  - `website/_toc.yml`
  - `website/intro.md`
  - `website/content/session_01_k_armed_bandit_blank.ipynb`
  - `.github/workflows/deploy-website.yml`

---

## If something breaks

- Website build complaining about Node.js:
  - Ensure `website/requirements.txt` still pins `jupyter-book<2`.

- Notebook import errors locally:
  - Install deps: `pip install -r requirements.txt`
  - Or use Docker and open JupyterLab.
