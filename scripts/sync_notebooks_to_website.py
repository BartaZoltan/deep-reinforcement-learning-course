#!/usr/bin/env python3
from pathlib import Path
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
    copied = 0

    for stem in stems:
        md_target = website_content / f"{stem}.md"
        src = sessions_root / stem / f"{stem}.ipynb"
        dst = website_content / f"{stem}.ipynb"

        # Non-notebook content page (e.g. source_references.md)
        if md_target.exists() and not src.exists():
            continue

        if not src.exists():
            missing.append(str(src.relative_to(repo_root)))
            continue

        shutil.copy2(src, dst)
        copied += 1

    print(f"Synced {copied} notebook(s) into website/content.")

    if missing:
        print("ERROR: Missing source notebook(s):")
        for m in missing:
            print(f"  - {m}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
