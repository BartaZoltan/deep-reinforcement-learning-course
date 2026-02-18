#!/usr/bin/env python3
import base64
from pathlib import Path
import json
import re
import sys

TASK_HEADING_PATTERN = re.compile(r"^\s*###\s*Task\s+\d+\b")


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
    missing_exercise = []
    generated = 0

    for stem in stems:
        md_target = website_content / f"{stem}.md"
        dst = website_content / f"{stem}.ipynb"
        session_dir = sessions_root / stem
        src_default = session_dir / f"{stem}.ipynb"
        src_empty = session_dir / f"{stem}_empty.ipynb"
        src_student = session_dir / f"{stem}_student.ipynb"

        # Non-notebook content page (e.g. source_references.md)
        if md_target.exists() and not src_default.exists():
            continue

        if not src_default.exists():
            missing.append(str(src_default.relative_to(repo_root)))
            continue

        exercise_name = None
        if src_empty.exists():
            exercise_name = f"{stem}_empty.ipynb"
        elif src_student.exists():
            exercise_name = f"{stem}_student.ipynb"
        else:
            missing_exercise.append(str(src_empty.relative_to(repo_root)))
            continue

        _generate_web_notebook(src_default, dst)
        _rewrite_colab_link(dst, stem, exercise_name)
        generated += 1

    print(f"Synced {generated} notebook(s) into website/content.")

    if missing:
        print("ERROR: Missing source notebook(s):")
        for m in missing:
            print(f"  - {m}")
        return 1

    if missing_exercise:
        print("ERROR: Missing paired exercise notebook(s) (_empty or _student expected):")
        for m in missing_exercise:
            print(f"  - {m}")
        return 1

    return 0


def _rewrite_colab_link(notebook_path: Path, stem: str, exercise_name: str) -> None:
    """Point Colab button(s) to the session's exercise notebook in published copy."""
    try:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception:
        return

    # Match links like .../session_x.ipynb or .../session_x_web.ipynb
    pattern = re.compile(
        rf"(https://colab\.research\.google\.com/github/[^)\s]+/notebooks/sessions/{re.escape(stem)}/){re.escape(stem)}(?:_web|_empty|_student)?\.ipynb"
    )

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        new_src = pattern.sub(rf"\1{exercise_name}", src)
        if new_src != src:
            cell["source"] = [new_src]
            changed = True

    if changed:
        notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_task_markdown_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    source = cell.get("source", [])
    first_line = source[0] if source else ""
    return bool(TASK_HEADING_PATTERN.match(first_line))


def _decode_mime_blob(blob: object) -> bytes:
    if isinstance(blob, list):
        raw = "".join(blob)
    elif isinstance(blob, str):
        raw = blob
    else:
        raise TypeError("Unsupported image payload type in notebook output.")
    return base64.b64decode(raw)


def _extract_image_paths_from_output(
    output: dict,
    assets_dir: Path,
    filename_prefix: str,
    output_parent: Path,
) -> list[str]:
    data = output.get("data", {})
    if not isinstance(data, dict):
        return []

    saved_paths: list[str] = []
    mime_to_ext = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
    }

    for mime, ext in mime_to_ext.items():
        if mime not in data:
            continue
        assets_dir.mkdir(parents=True, exist_ok=True)
        asset_path = assets_dir / f"{filename_prefix}{ext}"
        asset_path.write_bytes(_decode_mime_blob(data[mime]))
        saved_paths.append(asset_path.relative_to(output_parent).as_posix())

    return saved_paths


def _generate_web_notebook(input_path: Path, output_path: Path) -> None:
    notebook = json.loads(input_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    filtered = [cell for cell in cells if not _is_task_markdown_cell(cell)]

    output_parent = output_path.parent
    session_assets = output_parent / "assets" / "web_outputs"
    notebook_stem = output_path.stem.replace("_web", "")

    if session_assets.exists():
        for old in session_assets.glob(f"{notebook_stem}_cell*_out*.*"):
            old.unlink()

    new_cells: list[dict] = []
    for cell_idx, cell in enumerate(filtered):
        new_cells.append(cell)
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", []) or []
        embedded_paths: list[str] = []
        for out_idx, output in enumerate(outputs):
            if output.get("output_type") not in {"display_data", "execute_result"}:
                continue
            prefix = f"{notebook_stem}_cell{cell_idx:03d}_out{out_idx:02d}"
            embedded_paths.extend(
                _extract_image_paths_from_output(
                    output=output,
                    assets_dir=session_assets,
                    filename_prefix=prefix,
                    output_parent=output_parent,
                )
            )

        cell["outputs"] = []
        cell["execution_count"] = None

        if embedded_paths:
            lines = ["<!-- Embedded output asset(s) -->\n"]
            for rel_path in embedded_paths:
                lines.append(f"![cell-output]({rel_path})\n")
            new_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": lines,
                }
            )

    notebook["cells"] = new_cells
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
