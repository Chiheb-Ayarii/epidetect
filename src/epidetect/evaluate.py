"""Matching detections with reference events (ground truth or expert markings).

Contents: matching (step 3), metrics, sub-group counts for simulations and bootstrap
confidence intervals (step 4).

Matching rule: a detection and a reference event match if their time intervals overlap
(optionally extended by ``tolerance_s`` on each side). Matching is **one-to-one**: each
reference event can be matched by at most one detection and vice versa, so a long detection
covering two events cannot count twice, and two detections of the same event count as one true
positive and one false positive. Among overlapping pairs, the largest overlaps are matched
first (greedy assignment).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def overlap_s(
    a_on: np.ndarray, a_off: np.ndarray, b_on: np.ndarray, b_off: np.ndarray
) -> np.ndarray:
    """Pairwise overlap duration (s) between intervals a (rows) and b (columns); 0 if disjoint."""
    start = np.maximum(a_on[:, None], b_on[None, :])
    stop = np.minimum(a_off[:, None], b_off[None, :])
    return np.clip(stop - start, 0.0, None)


def match_events(
    detections: pd.DataFrame, reference: pd.DataFrame, tolerance_s: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matching of detections to reference events by interval overlap.

    Args:
        detections: Columns ``onset_s`` and ``offset_s``.
        reference: Columns ``onset_s`` and ``offset_s``.
        tolerance_s: Each reference interval is widened by this amount on both sides.

    Returns:
        ``det_match`` (for each detection, the index position of its matched reference event,
        or -1) and ``ref_match`` (for each reference event, the matched detection position,
        or -1).
    """
    n_det, n_ref = len(detections), len(reference)
    det_match = np.full(n_det, -1)
    ref_match = np.full(n_ref, -1)
    if n_det == 0 or n_ref == 0:
        return det_match, ref_match
    ov = overlap_s(
        detections["onset_s"].to_numpy(float),
        detections["offset_s"].to_numpy(float),
        reference["onset_s"].to_numpy(float) - tolerance_s,
        reference["offset_s"].to_numpy(float) + tolerance_s,
    )
    # Touching intervals (overlap exactly 0) are not a match.
    pairs = np.argwhere(ov > 0)
    order = np.argsort(-ov[pairs[:, 0], pairs[:, 1]], kind="stable")
    for i, j in pairs[order]:
        if det_match[i] == -1 and ref_match[j] == -1:
            det_match[i], ref_match[j] = j, i
    return det_match, ref_match


def count_matches(
    detections: pd.DataFrame, reference: pd.DataFrame, tolerance_s: float = 0.0
) -> dict[str, int]:
    """True positives, false positives and false negatives after one-to-one matching."""
    det_match, ref_match = match_events(detections, reference, tolerance_s)
    tp = int((det_match >= 0).sum())
    return {"tp": tp, "fp": len(detections) - tp, "fn": len(reference) - tp}


# --- Metrics -----------------------------------------------------------------------------------


def precision_recall_f1(tp: float, fp: float, fn: float) -> dict[str, float]:
    """Precision, recall and F1 from counts; NaN when undefined (0/0).

    F1 = 2 TP / (2 TP + FP + FN), the harmonic mean of precision and recall. There is no
    accuracy: "true negatives" are undefined for event detection (how many non-HFOs are there
    in a minute of signal?), and any proxy would be dominated by the overwhelming majority of
    HFO-free time.
    """
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1}


def score_simulated_band(
    detections: pd.DataFrame, truth: pd.DataFrame, band: str, spike_margin_s: float = 0.01
) -> dict[str, int]:
    """Counts for one band of one simulation, split by event sub-group.

    Args:
        detections: Detections of any band (only ``band`` is used).
        truth: Simulator ground truth (``kind``, ``on_spike``, ``onset_s``, ``offset_s``).
        band: ``"ripple"`` or ``"fast_ripple"``.
        spike_margin_s: Isolated-spike intervals are widened by this margin to attribute false
            positives to them (a filtered spike rings slightly beyond its nominal extent).

    Returns:
        ``tp, fp, fn``; ``n_isolated, tp_isolated`` and ``n_on_spike, tp_on_spike`` (HFOs of
        the band without / with an underlying spike); ``fp_on_spike`` (false positives that
        overlap an isolated spike); ``n_spikes, spikes_hit`` (isolated spikes overlapped by at
        least one false positive of this band).
    """
    det = detections[detections["band"] == band].reset_index(drop=True)
    ref = truth[truth["kind"] == band].reset_index(drop=True)
    det_match, ref_match = match_events(det, ref)
    found = ref_match >= 0
    on_spike = ref["on_spike"].to_numpy(bool)
    false_pos = det[det_match < 0]
    spikes = truth[truth["kind"] == "spike"]
    hit = (
        overlap_s(
            false_pos["onset_s"].to_numpy(float),
            false_pos["offset_s"].to_numpy(float),
            spikes["onset_s"].to_numpy(float) - spike_margin_s,
            spikes["offset_s"].to_numpy(float) + spike_margin_s,
        )
        > 0
    )
    return {
        "tp": int(found.sum()),
        "fp": int(len(false_pos)),
        "fn": int((~found).sum()),
        "n_isolated": int((~on_spike).sum()),
        "tp_isolated": int((found & ~on_spike).sum()),
        "n_on_spike": int(on_spike.sum()),
        "tp_on_spike": int((found & on_spike).sum()),
        "fp_on_spike": int(hit.any(axis=1).sum()) if hit.size else 0,
        "n_spikes": int(len(spikes)),
        "spikes_hit": int(hit.any(axis=0).sum()) if hit.size else 0,
    }


def bootstrap_ci(
    units: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap over independent units (rows of ``units``).

    Each replicate draws ``len(units)`` rows with replacement and recomputes ``statistic``.
    The unit must be the independent sampling unit: a patient for real data (events of one
    patient are correlated), a simulated channel for simulations.

    Returns:
        ``(estimate, low, high)``: the statistic on the original units and the
        ``(1 - level) / 2`` and ``(1 + level) / 2`` percentiles of the replicates (NaN
        replicates, e.g. 0/0, are ignored).
    """
    estimate = float(statistic(units))
    n = len(units)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    values = np.array([statistic(units.iloc[rows]) for rows in draws], dtype=float)
    if np.all(np.isnan(values)):
        return estimate, float("nan"), float("nan")
    alpha = (1 - level) / 2
    low, high = np.nanquantile(values, [alpha, 1 - alpha])
    return estimate, float(low), float(high)


def pooled(
    metric: str, tp: str = "tp", fp: str = "fp", fn: str = "fn"
) -> Callable[[pd.DataFrame], float]:
    """Statistic for ``bootstrap_ci``: a micro-averaged metric on summed counts."""

    def statistic(units: pd.DataFrame) -> float:
        totals = units[[tp, fp, fn]].sum()
        return precision_recall_f1(totals[tp], totals[fp], totals[fn])[metric]

    return statistic


def ratio(numerator: str, denominator: str) -> Callable[[pd.DataFrame], float]:
    """Statistic for ``bootstrap_ci``: sum(numerator) / sum(denominator), NaN if 0/0."""

    def statistic(units: pd.DataFrame) -> float:
        den = units[denominator].sum()
        return float(units[numerator].sum() / den) if den > 0 else float("nan")

    return statistic


def macro(column: str) -> Callable[[pd.DataFrame], float]:
    """Statistic for ``bootstrap_ci``: mean of a per-unit column, ignoring NaN units."""

    def statistic(units: pd.DataFrame) -> float:
        values = units[column].to_numpy(float)
        return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")

    return statistic
