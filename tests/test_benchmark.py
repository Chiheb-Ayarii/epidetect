"""Tests for threshold tuning helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epidetect import benchmark


def events(*pairs: tuple[float, float], channel: str | None = None) -> pd.DataFrame:
    table = pd.DataFrame(pairs, columns=["onset_s", "offset_s"])
    if channel is not None:
        table["channel"] = channel
    return table


def test_best_on_grid_takes_first_maximum() -> None:
    assert benchmark.best_on_grid([1, 2, 3, 4], {1: 0.2, 2: 0.6, 3: 0.6, 4: 0.1}.get) == (2.0, 0.6)


def test_count_by_channel_never_matches_across_channels() -> None:
    reference = pd.concat([events((1.0, 1.02), channel="A"), events((2.0, 2.02), channel="B")])
    detections = pd.concat([events((1.0, 1.02), channel="B"), events((2.0, 2.02), channel="B")])
    assert benchmark.count_by_channel(detections, reference) == {"tp": 1, "fp": 1, "fn": 1}


def test_tune_wavelet_filters_by_score() -> None:
    reference = events((1.0, 1.02), (2.0, 2.02))
    wavelet = events((1.0, 1.02), (2.0, 2.02), (3.0, 3.02)).assign(score=[12.0, 9.0, 6.0])
    unit = benchmark.TrainingUnit("u", reference, {}, wavelet, pd.DataFrame())
    # z=5 keeps the false positive (F1 0.8), z=8 keeps both true events only (F1 1.0).
    assert benchmark.tune_wavelet([unit], [5.0, 8.0, 10.0]) == (8.0, 1.0)


def test_tune_ml_threshold_counts_missed_references() -> None:
    table = pd.DataFrame({"label": [1, 1, 0, 0]})
    probabilities = np.array([0.9, 0.6, 0.7, 0.1])
    # 3 references, one of them never became a candidate.
    threshold, f1 = benchmark.tune_ml_threshold(table, probabilities, 3, [0.5, 0.65, 0.8])
    assert threshold == 0.5 and f1 == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_tune_band_and_detect_on_small_simulations() -> None:
    from epidetect import simulate
    from epidetect.detectors.energy import EnergyParams
    from epidetect.detectors.ml import MLParams
    from epidetect.detectors.wavelet import WaveletParams

    bands = {"ripple": (80.0, 250.0), "fast_ripple": (250.0, 500.0)}
    settings = benchmark.DetectorSettings(
        bands=bands,
        energy=EnergyParams(),
        wavelet=WaveletParams(),
        ml=MLParams(n_estimators=40),
        edge_margin_s=1.0,
        energy_grid=(3.0, 5.0),
        wavelet_grid=(6.0, 8.0),
        probability_grid=(0.3, 0.5, 0.7),
        inner_folds=3,
    )
    params = simulate.SimulationParams(duration_s=60.0).replace(events={"snr_db": 12.0})

    def unit(seed: int) -> benchmark.TrainingUnit:
        sim = simulate.simulate_from_seed(params, seed)
        analysed = benchmark.analyse_signal(sim.signal_uv, sim.fs, sim.events, settings)
        return benchmark.merge_channels({"sim": analysed["fast_ripple"]}, str(seed))

    tuned = benchmark.tune_band([unit(s) for s in (1, 2, 3)], "fast_ripple", settings)
    assert tuned.energy_k in (3.0, 5.0) and tuned.wavelet_z in (6.0, 8.0)
    assert tuned.ml_threshold in (0.3, 0.5, 0.7)
    test = unit(99)
    detections = benchmark.detect_tuned(tuned, test)
    assert set(detections) == {"energy", "wavelet", "ml"}
    f1 = benchmark.pooled_f1([benchmark.count_by_channel(detections["wavelet"], test.reference)])
    assert f1 > 0.5
