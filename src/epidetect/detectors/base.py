"""Shared detector utilities: output format, threshold segments, multi-channel loop.

Every detector returns a DataFrame with one row per detection and at least the columns of
``DETECTION_COLUMNS``:

* ``onset_s``, ``offset_s``: detection interval (s, from the start of the signal);
* ``peak_s``: time of maximal evidence (envelope or time-frequency peak);
* ``band``: ``"ripple"`` or ``"fast_ripple"``;
* ``freq_hz``: estimated oscillation frequency;
* ``score``: detector-specific confidence (higher = more confident), used for ROC-type analyses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

DETECTION_COLUMNS = ["onset_s", "offset_s", "peak_s", "band", "freq_hz", "score"]
BAND_NAMES = ("ripple", "fast_ripple")


def bands_from_config(det_cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """``{"ripple": (80, 250), "fast_ripple": (250, 500)}`` from ``[detectors.bands]``."""
    return {name: (float(low), float(high)) for name, (low, high) in det_cfg["bands"].items()}


def empty_detections(extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """An empty detection table with the standard columns."""
    return pd.DataFrame(columns=[*DETECTION_COLUMNS, *extra_columns])


def mask_to_segments(mask: np.ndarray) -> np.ndarray:
    """Runs of True in a boolean array as ``(start, stop)`` index pairs (stop exclusive)."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return changes.reshape(-1, 2)


def merge_segments(segments: np.ndarray, max_gap: int) -> np.ndarray:
    """Merge segments separated by at most ``max_gap`` samples."""
    if len(segments) == 0:
        return segments.reshape(0, 2)
    merged = [list(segments[0])]
    for start, stop in segments[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    return np.array(merged, dtype=int)


def zero_crossing_frequency(y: np.ndarray, fs: float) -> float:
    """Oscillation frequency from zero crossings: (crossings / 2) / duration."""
    crossings = np.count_nonzero(np.diff(np.signbit(y)))
    duration = len(y) / fs
    return float(crossings / 2 / duration) if duration > 0 else float("nan")


def detect_channels(
    detect: Callable[..., pd.DataFrame],
    data_uv: np.ndarray,
    fs: float,
    ch_names: list[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """Run a single-channel detector on every channel and add a ``channel`` column."""
    tables = []
    for name, x in zip(ch_names, data_uv, strict=True):
        table = detect(x, fs, **kwargs)
        table.insert(0, "channel", name)
        tables.append(table)
    if not tables:
        return empty_detections(("channel",))
    return pd.concat(tables, ignore_index=True)
