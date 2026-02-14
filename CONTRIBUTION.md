# Contribution Guide

This document explains how to add and maintain new course modules in this repository.

## Core Principle

Keep a single source of truth under `notebooks/sessions/`.
The website copies are generated automatically into `website/content/`.

## Session Structure

Create a folder per module:

- `notebooks/sessions/session_xx_topic/`

Recommended notebook files inside each session folder:

1. `session_xx_topic.ipynb`
- Main runnable notebook (developer/instructor version)
- Contains code and explanations

2. `session_xx_topic_web.ipynb`
- Website-focused version (recommended when you want curated visuals/storyline)
- Used by website sync with priority over the main notebook

3. `session_xx_topic_empty.ipynb` (required for published modules)
- Exercise/student version
- Colab button on website is automatically redirected to this file

Optional:

4. `session_xx_topic_homework.ipynb`
- Optional homework notebook

## Assets and Artifacts

Use this convention in each session folder:

- `assets/`
  - Stable images/files referenced by notebooks and website
  - Use this for final figures, diagrams, GIFs

- `artifacts/`
  - Temporary/generated outputs (intermediate plots, logs, model checkpoints)
  - Not meant to be long-term website references

Shared assets:

- `notebooks/shared_assets/`
  - Shared logo/template/buttons reused across modules

## Naming Convention

Use consistent session names everywhere:

- Folder: `session_xx_topic`
- Notebooks: `session_xx_topic.ipynb`, `session_xx_topic_web.ipynb`, `session_xx_topic_empty.ipynb`

`xx` is zero-padded (e.g. `01`, `02`, ..., `14`).

## Website Publishing Rules

Website navigation is controlled by:

- `website/_toc.yml`

For each entry like:

- `content/session_xx_topic`

the sync script will:

1. Look in `notebooks/sessions/session_xx_topic/`
2. Prefer `session_xx_topic_web.ipynb` if it exists
3. Otherwise use `session_xx_topic.ipynb`
4. Require `session_xx_topic_empty.ipynb`
5. Copy selected notebook to `website/content/session_xx_topic.ipynb`
6. Rewrite Colab link to `session_xx_topic_empty.ipynb`

## Sync and Build Commands

From repo root:

```bash
python scripts/sync_notebooks_to_website.py
jupyter-book build website
```

Use these before committing website changes.

## Colab Link Behavior

In published website notebooks, the Colab button is rewritten automatically to the `_empty` notebook.
No manual editing is required in the website copy.

## Images in Notebooks

Prefer absolute raw GitHub URLs when you need robust rendering in web/Colab contexts.
Typical pattern:

```text
https://raw.githubusercontent.com/<org-or-user>/<repo>/main/notebooks/sessions/session_xx_topic/assets/<file>
```

This avoids broken relative paths after conversion/publishing.

## Creating a New Module (Checklist)

1. Create folder `notebooks/sessions/session_xx_topic/`
2. Add `session_xx_topic.ipynb`
3. Add `session_xx_topic_empty.ipynb`
4. Optionally add `session_xx_topic_web.ipynb`
5. Create `assets/` and put final figures there
6. Add module to `website/_toc.yml` as `content/session_xx_topic`
7. Run sync + build
8. Validate locally in `website/_build/html/`

## Quality Checks Before Commit

- Notebook names/folder names match the `session_xx_topic` pattern
- `_empty` exists for every module in website TOC
- No broken image links
- Website builds successfully
- Generated website notebooks are not edited manually

## Important Do/Don't

Do:

- Edit only notebooks in `notebooks/sessions/`
- Treat `website/content/*.ipynb` as generated files
- Keep stable visuals in `assets/`

Don't:

- Maintain separate manual copies in `website/content/`
- Depend on heavy runtime execution in GitHub Actions for plot generation
- Store temporary training outputs in `assets/`

## Existing Automation

Deployment workflow:

- `.github/workflows/deploy-website.yml`

Sync script:

- `scripts/sync_notebooks_to_website.py`

These already implement the publishing flow described above.
