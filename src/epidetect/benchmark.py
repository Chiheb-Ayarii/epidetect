"""Threshold tuning shared by the simulation and real-data benchmarks.

Rules that prevent leakage:

* every threshold is chosen on **training units only** (simulated channels or patients);
* the ML probability threshold is chosen on **out-of-fold** predictions: the training units are
  split into groups (GroupKFold), each group is predicted by a forest trained on the others,
  so the threshold is never chosen on predictions of a model that saw the same data.

Grid choice: the value with the highest pooled F1; among ties, the first value of the grid.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from epidetect import evaluate
from epidetect.detectors.ml import MLDetector


@dataclass(frozen=True)
class TrainingUnit:
    """Detector outputs and reference of one training unit, in one band.

    Attributes:
        unit: Identifier (simulation seed or patient).
        reference: Reference events of the band (``onset_s``, ``offset_s``, optional
            ``channel``).
        energy: Energy detections for every grid value ``k`` (dict ``k -> table``).
        wavelet: Wavelet detections at the lowest grid threshold (filtered later by ``score``).
        candidates: ML candidate table with features and a ``label`` column.
    """

    unit: str
    reference: pd.DataFrame
    energy: dict[float, pd.DataFrame]
    wavelet: pd.DataFrame
    candidates: pd.DataFrame


def count_by_channel(detections: pd.DataFrame, reference: pd.DataFrame) -> dict[str, int]:
    """TP / FP / FN with matching done channel by channel (if a ``channel`` column exists)."""
    if "channel" not in reference.columns and "channel" not in detections.columns:
        return evaluate.count_matches(detections, reference)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    channels = set(detections.get("channel", pd.Series(dtype=str))) | set(
        reference.get("channel", pd.Series(dtype=str))
    )
    for channel in channels:
        counts = evaluate.count_matches(
            detections[detections["channel"] == channel], reference[reference["channel"] == channel]
        )
        for key in totals:
            totals[key] += counts[key]
    return totals


def best_on_grid(grid: Sequence[float], f1_of: Callable[[float], float]) -> tuple[float, float]:
    """Grid value with the highest F1 (first one among ties) and that F1."""
    scores = [f1_of(value) for value in grid]
    best = int(np.nanargmax(scores))
    return float(grid[best]), float(scores[best])


def pooled_f1(counts: Sequence[dict[str, int]]) -> float:
    """Micro-averaged F1 over several units."""
    tp, fp, fn = (sum(c[k] for c in counts) for k in ("tp", "fp", "fn"))
    return evaluate.precision_recall_f1(tp, fp, fn)["f1"]


def tune_energy(units: Sequence[TrainingUnit], grid: Sequence[float]) -> tuple[float, float]:
    """Energy threshold k maximising the pooled F1 over the training units."""
    return best_on_grid(
        grid, lambda k: pooled_f1([count_by_channel(u.energy[k], u.reference) for u in units])
    )


def tune_wavelet(units: Sequence[TrainingUnit], grid: Sequence[float]) -> tuple[float, float]:
    """Wavelet z threshold maximising the pooled F1 (detections filtered by ``score``)."""

    def f1_of(z: float) -> float:
        return pooled_f1(
            [count_by_channel(u.wavelet[u.wavelet["score"] >= z], u.reference) for u in units]
        )

    return best_on_grid(grid, f1_of)


def out_of_fold_probabilities(
    table: pd.DataFrame, groups: np.ndarray, make_detector: Callable[[], MLDetector], n_splits: int
) -> np.ndarray:
    """Probability of each candidate predicted by a forest trained on the other groups."""
    probabilities = np.full(len(table), np.nan)
    n_splits = min(n_splits, len(np.unique(groups)))
    for train_rows, test_rows in GroupKFold(n_splits=n_splits).split(table, groups=groups):
        detector = make_detector().fit_table(table.iloc[train_rows])
        scored = detector.predict_table(table.iloc[test_rows])
        probabilities[test_rows] = scored["score"].to_numpy(float)
    return probabilities


def tune_ml_threshold(
    table: pd.DataFrame,
    probabilities: np.ndarray,
    n_reference: int,
    grid: Sequence[float],
) -> tuple[float, float]:
    """Probability threshold maximising the pooled F1 of out-of-fold predictions.

    Args:
        table: Candidates with a ``label`` column (1 = matches a reference event).
        probabilities: Out-of-fold probabilities (NaN = band not learnable in that fold,
            treated as 0).
        n_reference: Number of reference events, *including* those never captured by any
            candidate: they are false negatives whatever the threshold.
    """
    labels = table["label"].to_numpy(int)
    p = np.nan_to_num(probabilities, nan=0.0)

    def f1_of(threshold: float) -> float:
        predicted = p >= threshold
        tp = int((predicted & (labels == 1)).sum())
        fp = int((predicted & (labels == 0)).sum())
        return evaluate.precision_recall_f1(tp, fp, n_reference - tp)["f1"]

    return best_on_grid(grid, f1_of)
