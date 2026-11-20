"""Tests for the one-to-one matching of detections and reference events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epidetect import evaluate


def intervals(*pairs: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["onset_s", "offset_s"])


def test_counts_hand_made_example() -> None:
    reference = intervals((1.0, 1.02), (2.0, 2.03), (5.0, 5.01))
    detections = intervals(
        (1.01, 1.04),  # overlaps event 0 -> TP
        (2.02, 2.05),  # overlaps event 1 -> TP
        (2.025, 2.06),  # second detection of event 1 -> FP (one-to-one)
        (3.0, 3.02),  # nothing there -> FP
    )
    assert evaluate.count_matches(detections, reference) == {"tp": 2, "fp": 2, "fn": 1}


def test_one_detection_cannot_match_two_events() -> None:
    reference = intervals((1.0, 1.01), (1.02, 1.03))
    detections = intervals((0.99, 1.04))
    assert evaluate.count_matches(detections, reference) == {"tp": 1, "fp": 0, "fn": 1}


def test_largest_overlap_wins() -> None:
    reference = intervals((1.00, 1.02), (1.03, 1.05))
    detections = intervals((1.015, 1.045))  # 5 ms with event 0, 15 ms with event 1
    det_match, ref_match = evaluate.match_events(detections, reference)
    assert det_match.tolist() == [1] and ref_match.tolist() == [-1, 0]


def test_touching_is_not_overlap_but_tolerance_helps() -> None:
    reference = intervals((1.0, 1.02))
    detections = intervals((1.02, 1.03))
    assert evaluate.count_matches(detections, reference)["tp"] == 0
    assert evaluate.count_matches(detections, reference, tolerance_s=0.005)["tp"] == 1


def test_empty_inputs() -> None:
    empty = intervals()
    ref = intervals((1.0, 1.1))
    assert evaluate.count_matches(empty, ref) == {"tp": 0, "fp": 0, "fn": 1}
    assert evaluate.count_matches(ref, empty) == {"tp": 0, "fp": 1, "fn": 0}
    det_match, ref_match = evaluate.match_events(empty, empty)
    assert det_match.size == 0 and ref_match.size == 0 and np.all(det_match == -1)
