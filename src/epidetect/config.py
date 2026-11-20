"""Load the project configuration (TOML) and resolve paths against the repository root."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read a TOML configuration file.

    Args:
        path: Configuration file. Defaults to ``configs/default.toml``.

    Returns:
        The configuration as nested dictionaries. Entries of the ``[paths]`` section are
        converted to absolute :class:`pathlib.Path` objects relative to the repository root.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    config["paths"] = {key: REPO_ROOT / value for key, value in config.get("paths", {}).items()}
    return config
