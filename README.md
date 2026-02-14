# Deep Reinforcement Learning Course

Repository structure:

- `website/`: static course website (Jupyter Book source and build artifacts)
- `notebooks/sessions/`: source Jupyter notebooks organized by session folder (`session_XX_*`)
- `notebooks/shared_assets/`: shared notebook assets (logo, template, buttons)
- `notes/`: course planning and internal notes

## Quick navigation

- Course plan: `notes/Course_plan.md`
- Continuation notes: `notes/CONTINUATION_NOTES.md`
- Session notebooks: `notebooks/sessions/`
- Website docs: `website/README.md`

## Build website

From repository root:

```bash
pip install -r website/requirements.txt
jupyter-book build website
```
