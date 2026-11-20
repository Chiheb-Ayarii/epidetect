"""Matching detections with reference events (ground truth or expert markings).

Step 3 only needs the matching (to label ML training candidates); metrics, per-class and
per-SNR evaluation and bootstrap confidence intervals are added in step 4.

Matching rule: a detection and a reference event match if their time intervals overlap
(optionally extended by ``tolerance_s`` on each side). Matching is **one-to-one**: each
reference event can be matched by at most one detection and vice versa, so a long detection
covering two events cannot count twice, and two detections of the same event count as one true
positive and one false positive. Among overlapping pairs, the largest overlaps are matched
first (greedy assignment).
"""

from __future__ import annotations

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
