from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IGNORED_FILENAMES = {".gitkeep"}


def clear_notebook_outputs(notebook_path: Path) -> None:
    """Remove saved outputs and execution counts from one notebook."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def clean_project(
    project_root: str | Path,
    clear_outputs: bool = True,
) -> None:
    """Remove generated files, caches, checkpoints, and notebook outputs."""
    project_root = Path(project_root).resolve()

    folders_to_remove = [
        folder
        for folder_name in [".ipynb_checkpoints", "__pycache__"]
        for folder in project_root.rglob(folder_name)
        if ".venv" not in folder.parts
    ]

    for folder in folders_to_remove:
        if folder.exists():
            shutil.rmtree(folder)
            print("Removed folder:", folder.relative_to(project_root))

    generated_folders = [
        project_root / "data" / "processed",
        project_root / "outputs",
    ]

    for folder in generated_folders:
        if not folder.exists():
            continue

        for path in folder.iterdir():
            if path.name in IGNORED_FILENAMES or path.name == "private":
                continue

            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

            print("Removed:", path.relative_to(project_root))

    if clear_outputs:
        for notebook_path in (project_root / "notebooks").glob("*.ipynb"):
            clear_notebook_outputs(notebook_path)
            print("Cleared notebook:", notebook_path.relative_to(project_root))

    (project_root / "data" / "processed").mkdir(
        parents=True,
        exist_ok=True,
    )
    (project_root / "data" / "processed" / ".gitkeep").touch()

    (project_root / "outputs").mkdir(
        parents=True,
        exist_ok=True,
    )
    (project_root / "outputs" / ".gitkeep").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean generated files before committing the project.",
    )
    parser.add_argument(
        "project_root",
        nargs="?",
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--keep-notebook-outputs",
        action="store_true",
    )

    args = parser.parse_args()

    clean_project(
        project_root=args.project_root,
        clear_outputs=not args.keep_notebook_outputs,
    )
