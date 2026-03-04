#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path


TASK_HEADING_PATTERN = re.compile(r"^\s*###\s*Task\s+\d+\b")
DELIMITER_LINE = "#" * 72
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/BartaZoltan/deep-reinforcement-learning-course/main"
CELL_OUTPUT_FILE_PATTERN = re.compile(
    r"CELL_OUTPUT_DIR\s*/\s*['\"]([^'\"]+\.(?:gif|png|jpg|jpeg))['\"]",
    flags=re.IGNORECASE,
)
CELL_OUTPUT_SUBDIR_PATTERN = re.compile(
    r"output_dir\s*=\s*CELL_OUTPUT_DIR\s*/\s*['\"]([^'\"]+)['\"]"
)
CELL_OUTPUT_GIF_PATTERN = re.compile(
    r"gif_name\s*=\s*['\"]([^'\"]+\.(?:gif|png|jpg|jpeg))['\"]",
    flags=re.IGNORECASE,
)
MARKDOWN_ASSET_LINK_PATTERN = re.compile(
    r"(!?\[[^\]]*\]\()(?P<path>assets/[^)\s]+)(\))"
)
HTML_ASSET_ATTR_PATTERN = re.compile(
    r"(\b(?:src|href)=['\"])(?P<path>assets/[^'\"]+)(['\"])"
)


def _is_task_markdown_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    first_line = ""
    source = cell.get("source", [])
    if source:
        first_line = source[0]
    return bool(TASK_HEADING_PATTERN.match(first_line))


def _default_output_for(input_path: Path) -> Path:
    if input_path.suffix != ".ipynb":
        return input_path.with_name(input_path.name + "_web")
    stem = input_path.stem
    if stem.endswith("_web"):
        return input_path
    return input_path.with_name(f"{stem}_web.ipynb")


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


def _github_raw_url_for(rel_path: str, source_parent: Path) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    src = source_parent / rel_path
    if not src.exists():
        return None
    try:
        repo_rel = src.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None
    return f"{GITHUB_RAW_BASE}/{repo_rel}"


def _source_parent_raw_base(source_parent: Path) -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        rel_parent = source_parent.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None
    return f"{GITHUB_RAW_BASE}/{rel_parent}"


def _rewrite_markdown_asset_links(cell: dict, source_parent: Path) -> None:
    if cell.get("cell_type") != "markdown":
        return

    raw_base = _source_parent_raw_base(source_parent)
    if raw_base is None:
        return

    source = "".join(cell.get("source", []) or [])
    if "assets/" not in source:
        return

    def _markdown_repl(match: re.Match[str]) -> str:
        path = match.group("path")
        return f"{match.group(1)}{raw_base}/{path}{match.group(3)}"

    def _html_repl(match: re.Match[str]) -> str:
        path = match.group("path")
        return f"{match.group(1)}{raw_base}/{path}{match.group(3)}"

    rewritten = MARKDOWN_ASSET_LINK_PATTERN.sub(_markdown_repl, source)
    rewritten = HTML_ASSET_ATTR_PATTERN.sub(_html_repl, rewritten)
    cell["source"] = [rewritten]


def _extract_saved_asset_paths_from_source(
    cell: dict,
    output_parent: Path,
    source_parent: Path,
) -> list[str]:
    if cell.get("cell_type") != "code":
        return []

    source = "".join(cell.get("source", []) or [])
    found: list[str] = []

    for rel_file in CELL_OUTPUT_FILE_PATTERN.findall(source):
        rel_path = f"assets/cell_outputs/{rel_file}"
        raw_url = _github_raw_url_for(rel_path=rel_path, source_parent=source_parent)
        if raw_url:
            found.append(raw_url)

    subdirs = CELL_OUTPUT_SUBDIR_PATTERN.findall(source)
    gif_names = CELL_OUTPUT_GIF_PATTERN.findall(source)
    for subdir in subdirs:
        for gif_name in gif_names:
            rel_path = f"assets/cell_outputs/{subdir}/{gif_name}"
            raw_url = _github_raw_url_for(rel_path=rel_path, source_parent=source_parent)
            if raw_url:
                found.append(raw_url)

    return list(dict.fromkeys(found))


def generate_web_notebook(input_path: Path, output_path: Path) -> tuple[int, int]:
    notebook = json.loads(input_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    filtered = [cell for cell in cells if not _is_task_markdown_cell(cell)]
    removed = len(cells) - len(filtered)

    output_parent = output_path.parent
    session_assets = output_parent / "assets" / "web_outputs"
    notebook_stem = output_path.stem.replace("_web", "")

    # Remove stale generated assets for this notebook.
    if session_assets.exists():
        for old in session_assets.glob(f"{notebook_stem}_cell*_out*.*"):
            old.unlink()

    new_cells: list[dict] = []
    for cell_idx, cell in enumerate(filtered):
        _rewrite_markdown_asset_links(cell=cell, source_parent=input_path.parent)
        new_cells.append(cell)
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", []) or []
        cell["source"] = [line for line in source if line.strip() != DELIMITER_LINE]

        source_asset_paths = _extract_saved_asset_paths_from_source(
            cell=cell,
            output_parent=output_parent,
            source_parent=input_path.parent,
        )
        if source_asset_paths:
            embedded_paths = source_asset_paths
        else:
            outputs = cell.get("outputs", []) or []
            embedded_paths = []
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

        # Clear runtime outputs from generated web notebook.
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
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return removed, len(new_cells) - len(filtered)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate web notebook by removing markdown cells that start with '### Task n'."
    )
    parser.add_argument("input", type=Path, help="Path to source .ipynb notebook.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to generated web notebook (default: <input>_web.ipynb).",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output if args.output is not None else _default_output_for(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input notebook does not exist: {input_path}")

    removed, inserted = generate_web_notebook(input_path, output_path)
    print(f"Generated: {output_path}")
    print(f"Removed task markdown cells: {removed}")
    print(f"Inserted embedded-output markdown cells: {inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
