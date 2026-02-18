#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DELIMITER = "#" * 72


def _is_delimiter_line(line: str, delimiter: str) -> bool:
    return line.strip() == delimiter


def _blank_like(line: str) -> str:
    return "\n" if line.endswith("\n") else ""


def _strip_between_delimiters(lines: list[str], delimiter: str) -> tuple[list[str], int]:
    in_block = False
    blanked = 0
    output: list[str] = []

    for line in lines:
        if _is_delimiter_line(line, delimiter):
            in_block = not in_block
            output.append(line)
            continue

        if in_block:
            output.append(_blank_like(line))
            blanked += 1
        else:
            output.append(line)

    if in_block:
        raise ValueError("Unmatched delimiter: found an opening delimiter without a closing pair.")

    return output, blanked


def generate_student_notebook(input_path: Path, output_path: Path, delimiter: str = DEFAULT_DELIMITER) -> tuple[int, int]:
    notebook = json.loads(input_path.read_text(encoding="utf-8"))
    code_cells = 0
    blanked_lines = 0

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = cell.get("source", [])
        new_source, changed = _strip_between_delimiters(source, delimiter)
        blanked_lines += changed
        cell["source"] = new_source
        # Student version should not include executed outputs.
        cell["outputs"] = []
        cell["execution_count"] = None

    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return code_cells, blanked_lines


def _default_output_for(input_path: Path) -> Path:
    if input_path.suffix != ".ipynb":
        return input_path.with_name(input_path.name + "_student")
    return input_path.with_name(f"{input_path.stem}_student.ipynb")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate student notebook by blanking code between delimiter lines."
    )
    parser.add_argument("input", type=Path, help="Path to source .ipynb notebook.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to generated student notebook (default: <input>_student.ipynb).",
    )
    parser.add_argument(
        "--delimiter",
        default=DEFAULT_DELIMITER,
        help=f"Exact delimiter line used as block markers (default: {DEFAULT_DELIMITER!r}).",
    )

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output if args.output is not None else _default_output_for(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input notebook does not exist: {input_path}")

    code_cells, blanked_lines = generate_student_notebook(
        input_path=input_path,
        output_path=output_path,
        delimiter=args.delimiter,
    )
    print(f"Generated: {output_path}")
    print(f"Processed code cells: {code_cells}")
    print(f"Blanked lines between delimiter pairs: {blanked_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
