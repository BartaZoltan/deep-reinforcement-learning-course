#!/usr/bin/env python3
from pathlib import Path
import json
import re
import shutil
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    toc_path = repo_root / "website" / "_toc.yml"
    website_content = repo_root / "website" / "content"
    sessions_root = repo_root / "notebooks" / "sessions"

    if not toc_path.exists():
        print(f"ERROR: Missing TOC file: {toc_path}")
        return 1

    toc_text = toc_path.read_text(encoding="utf-8")
    stems = re.findall(r"^\s*-\s*file:\s*content/([A-Za-z0-9_\-/]+)\s*$", toc_text, flags=re.MULTILINE)

    website_content.mkdir(parents=True, exist_ok=True)

    # Remove previously generated notebooks in website/content
    for p in website_content.glob("*.ipynb"):
        p.unlink()

    missing = []
    missing_empty = []
    copied = 0

    for stem in stems:
        md_target = website_content / f"{stem}.md"
        dst = website_content / f"{stem}.ipynb"
        session_dir = sessions_root / stem
        src_web = session_dir / f"{stem}_web.ipynb"
        src_default = session_dir / f"{stem}.ipynb"
        src_empty = session_dir / f"{stem}_empty.ipynb"

        # Non-notebook content page (e.g. source_references.md)
        if md_target.exists() and not src_default.exists() and not src_web.exists():
            continue

        # Prefer pre-rendered web notebook when available, fallback to default.
        src = src_web if src_web.exists() else src_default

        if not src.exists():
            missing.append(str(src_default.relative_to(repo_root)))
            continue

        # Enforce pairing with an exercise version.
        if not src_empty.exists():
            missing_empty.append(str(src_empty.relative_to(repo_root)))
            continue

        shutil.copy2(src, dst)
        _rewrite_colab_to_empty(dst, stem)
        copied += 1

    print(f"Synced {copied} notebook(s) into website/content.")

    if missing:
        print("ERROR: Missing source notebook(s):")
        for m in missing:
            print(f"  - {m}")
        return 1

    if missing_empty:
        print("ERROR: Missing paired _empty notebook(s):")
        for m in missing_empty:
            print(f"  - {m}")
        return 1

    return 0


def _rewrite_colab_to_empty(notebook_path: Path, stem: str) -> None:
    """Point Colab button(s) to the session's _empty notebook in published copy."""
    try:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception:
        return

    # Match links like .../session_x.ipynb or .../session_x_web.ipynb
    pattern = re.compile(
        rf"(https://colab\.research\.google\.com/github/[^)\s]+/notebooks/sessions/{re.escape(stem)}/){re.escape(stem)}(?:_web|_empty)?\.ipynb"
    )

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        new_src = pattern.sub(rf"\1{stem}_empty.ipynb", src)
        if new_src != src:
            cell["source"] = [new_src]
            changed = True

    if changed:
        notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
