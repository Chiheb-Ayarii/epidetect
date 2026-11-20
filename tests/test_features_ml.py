"""Tests for candidate features and the machine-learning detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_detectors_energy import BANDS, FS, NO_EVENTS, add_burst, background

from epidetect import evaluate, simulate
from epidetect.detectors.energy import detect_energy
from epidetect.detectors.ml import MLDetector, MLParams, label_candidates
from epidetect.features import FEATURE_NAMES, extract_features

SMALL_FOREST = MLParams(n_estimators=60)


def candidate(onset: float, offset: float, band: str) -> dict:
    return {
        "onset_s": onset,
        "offset_s": offset,
        "peak_s": (onset + offset) / 2,
        "band": band,
        "freq_hz": np.nan,
        "score": np.nan,
    }


def test_features_of_a_clean_burst_and_of_a_spike() -> None:
    x = background(30.0, 1)
    truth = add_burst(x, 10.0, 300.0, 8, snr_db=15, band=BANDS["fast_ripple"])
    spike = simulate.spike_waveform(FS, -900.0, 0.003, 0.008, 0.3, 0.06, 0.03, apex_sigma_s=0.0013)
    centre = int(20.0 * FS)
    x[centre - len(spike) // 2 : centre - len(spike) // 2 + len(spike)] += spike
    cands = pd.DataFrame(
        [
            candidate(truth.onset_s[0], truth.offset_s[0], "fast_ripple"),
            candidate(19.995, 20.005, "fast_ripple"),
        ]
    )
    f = extract_features(x, FS, cands, BANDS)
    assert list(f.columns) == FEATURE_NAMES
    hfo, spk = f.iloc[0], f.iloc[1]
    assert hfo.freq_hz == pytest.approx(300.0, rel=0.1)
    assert hfo.duration_ms == pytest.approx(8 / 300 * 1000, rel=0.05)
    assert hfo.tf_valley_ratio < 0.3 < spk.tf_valley_ratio
    for name in ("raw_amplitude_z", "sharpness_z", "low_band_ratio_db"):
        assert spk[name] > hfo[name]


def test_label_candidates_matches_within_band_only() -> None:
    table = pd.DataFrame(
        [
            candidate(1.00, 1.02, "fast_ripple"),
            candidate(2.00, 2.02, "ripple"),
            candidate(2.005, 2.03, "fast_ripple"),
        ]
    )
    reference = pd.DataFrame(
        {"onset_s": [1.01, 2.01], "offset_s": [1.03, 2.03], "kind": ["fast_ripple", "ripple"]}
    )
    assert label_candidates(table, reference).tolist() == [1, 1, 0]


@pytest.fixture(scope="module")
def trained() -> MLDetector:
    params = simulate.SimulationParams(duration_s=180.0).replace(
        events={"snr_db": 10.0, "snr_jitter_db": 4.0}
    )
    sims = [simulate.simulate_from_seed(params, seed) for seed in (101, 102)]
    return MLDetector(BANDS, SMALL_FOREST).fit([(s.signal_uv, s.fs, s.events) for s in sims])


def test_ml_detector_generalises_to_an_unseen_simulation(trained: MLDetector) -> None:
    test = simulate.simulate_from_seed(simulate.SimulationParams(duration_s=180.0), 999)
    det = trained.detect(test.signal_uv, test.fs)
    for band in BANDS:
        counts = evaluate.count_matches(
            det[det.band == band], test.events[test.events.kind == band]
        )
        f1 = 2 * counts["tp"] / (2 * counts["tp"] + counts["fp"] + counts["fn"])
        assert f1 > 0.6, (band, counts)


def test_ml_rejects_isolated_spikes_better_than_energy(trained: MLDetector) -> None:
    p = simulate.SimulationParams(duration_s=120.0).replace(
        events={**NO_EVENTS, "rate_spike_per_min": 12.0}
    )
    sim = simulate.simulate_from_seed(p, 7)
    assert len(trained.detect(sim.signal_uv, FS)) < len(detect_energy(sim.signal_uv, FS, BANDS))


def test_training_is_reproducible(trained: MLDetector) -> None:
    sim = simulate.simulate_from_seed(simulate.SimulationParams(duration_s=60.0), 5)
    table = trained.table(sim.signal_uv, FS)
    again = MLDetector(BANDS, SMALL_FOREST)
    again.models = trained.models
    pd.testing.assert_frame_equal(trained.predict_table(table), again.predict_table(table))
    assert trained.feature_importances().shape == (len(FEATURE_NAMES), 2)


def test_band_without_positives_gets_no_model() -> None:
    table = pd.DataFrame([candidate(1.0, 1.02, "ripple")] * 4)
    for name in FEATURE_NAMES:
        table[name] = np.arange(4.0)
    table["label"] = 0
    detector = MLDetector(BANDS, SMALL_FOREST).fit_table(table)
    assert detector.models == {}
    assert detector.detections_from_table(table.drop(columns="label")).empty
