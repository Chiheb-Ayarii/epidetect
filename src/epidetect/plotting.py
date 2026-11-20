"""Shared figure style so that every figure of the project looks the same.

Colours: neutral grey for ordinary data, one accent colour for what the reader must notice,
fixed categorical slots for named series (never cycled), a single-hue sequential map for
magnitudes. The categorical slots come from a colour-blind-validated palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only: no window, works in CI and scripts

import matplotlib.pyplot as plt  # noqa: E402

NEUTRAL = "#9a9893"  # ordinary channels / background data
ACCENT = "#eb6834"  # what the reader must notice (e.g. flagged channels)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BAND_RIPPLE = "#2a78d6"
BAND_FAST_RIPPLE = "#6250d6"
SEQUENTIAL_CMAP = "Blues"


def apply_style() -> None:
    """Thin marks, recessive grid and axes, readable fonts."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "lines.linewidth": 1.0,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    """Save a figure as PNG (creating the folder) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
