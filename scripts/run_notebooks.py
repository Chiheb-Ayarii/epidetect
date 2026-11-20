"""Execute every notebook in notebooks/ in place, so that saved outputs are always current.

Usage:
    python scripts/run_notebooks.py [notebooks/01_loading_and_qc.ipynb ...]

Notebooks need the real data (data/bids), so they are not run in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 1200  # per cell


def execute(path: Path) -> None:
    """Run one notebook top to bottom with the current interpreter's kernel and save it."""
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        record_timing=False,  # keep diffs clean: no timestamps in the saved file
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"executed {path.relative_to(REPO_ROOT)}")


def main() -> None:
    paths = [Path(p).resolve() for p in sys.argv[1:]] or sorted(
        (REPO_ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")
    )
    for path in paths:
        execute(path)


if __name__ == "__main__":
    main()
