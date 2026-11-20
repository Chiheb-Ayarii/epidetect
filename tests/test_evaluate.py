"""Tests for the one-to-one matching of detections and reference events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_precision_recall_f1_and_undefined_cases() -> None:
    m = evaluate.precision_recall_f1(tp=6, fp=2, fn=4)
    assert m["precision"] == 0.75 and m["recall"] == 0.6
    assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
    nothing = evaluate.precision_recall_f1(0, 0, 0)
    assert all(np.isnan(v) for v in nothing.values())
    no_reference = evaluate.precision_recall_f1(0, 3, 0)
    assert no_reference["precision"] == 0 and np.isnan(no_reference["recall"])


def test_score_simulated_band_subgroups() -> None:
    truth = pd.DataFrame(
        {
            "kind": ["fast_ripple", "fast_ripple", "fast_ripple", "spike", "ripple"],
            "on_spike": [False, True, False, False, False],
            "onset_s": [1.0, 2.0, 3.0, 4.0, 5.0],
            "offset_s": [1.02, 2.02, 3.02, 4.01, 5.03],
        }
    )
    detections = pd.DataFrame(
        {
            "band": ["fast_ripple", "fast_ripple", "fast_ripple", "ripple", "fast_ripple"],
            "onset_s": [1.01, 2.01, 4.012, 5.0, 6.0],  # FR iso, FR on spike, ~spike, ripple, noise
            "offset_s": [1.03, 2.03, 4.02, 5.02, 6.01],
        }
    )
    c = evaluate.score_simulated_band(detections, truth, "fast_ripple", spike_margin_s=0.005)
    assert (c["tp"], c["fp"], c["fn"]) == (2, 2, 1)
    assert (c["n_isolated"], c["tp_isolated"], c["n_on_spike"], c["tp_on_spike"]) == (2, 1, 1, 1)
    assert (c["fp_on_spike"], c["n_spikes"], c["spikes_hit"]) == (1, 1, 1)


def test_bootstrap_ci_brackets_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    units = pd.DataFrame(
        {"tp": rng.integers(5, 15, 20), "fp": rng.integers(0, 5, 20), "fn": rng.integers(0, 5, 20)}
    )
    est, low, high = evaluate.bootstrap_ci(units, evaluate.pooled("f1"), n_boot=500, seed=3)
    assert low <= est <= high and high - low < 0.2
    assert evaluate.bootstrap_ci(units, evaluate.pooled("f1"), n_boot=500, seed=3) == (
        est,
        low,
        high,
    )
    same = pd.DataFrame({"tp": [5] * 10, "fp": [1] * 10, "fn": [2] * 10})
    est, low, high = evaluate.bootstrap_ci(same, evaluate.pooled("recall"), n_boot=200)
    assert est == low == high == pytest.approx(5 / 7)


def test_macro_and_ratio_statistics_ignore_undefined_units() -> None:
    units = pd.DataFrame({"f1": [0.5, np.nan, 1.0], "hit": [1, 0, 3], "n": [2, 0, 3]})
    assert evaluate.macro("f1")(units) == 0.75
    assert evaluate.ratio("hit", "n")(units) == 0.8
