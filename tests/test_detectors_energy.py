"""Tests for the RMS energy detector and the shared detector utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epidetect import evaluate, filters, simulate
from epidetect.detectors import base
from epidetect.detectors.energy import EnergyParams, detect_energy
from epidetect.simulate import BackgroundParams

FS = 2000.0
BANDS = {"ripple": (80.0, 250.0), "fast_ripple": (250.0, 500.0)}
NO_EVENTS = {"rate_ripple_per_min": 0.0, "rate_fast_ripple_per_min": 0.0, "rate_spike_per_min": 0.0}


def background(duration_s: float, seed: int) -> np.ndarray:
    bg = BackgroundParams()
    n = int(duration_s * FS)
    return bg.std_uv * simulate.colored_noise(n, FS, bg, np.random.default_rng(seed))


def add_burst(
    x: np.ndarray,
    centre_s: float,
    freq: float,
    n_cycles: float,
    snr_db: float,
    band: tuple[float, float],
) -> pd.DataFrame:
    """Add an HFO with a given in-band SNR; return its ground-truth interval."""
    unit = simulate.hfo_waveform(FS, freq, n_cycles)
    band_power = filters.bandpass(x, FS, band)[int(FS) : -int(FS)].var()
    amplitude = np.sqrt(10 ** (snr_db / 10) * band_power / np.mean(unit**2))
    centre = int(centre_s * FS)
    x[centre - len(unit) // 2 : centre - len(unit) // 2 + len(unit)] += amplitude * unit
    half = n_cycles / freq / 2
    return pd.DataFrame({"onset_s": [centre_s - half], "offset_s": [centre_s + half]})


def test_mask_to_segments_and_merge() -> None:
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1], dtype=bool)
    segments = base.mask_to_segments(mask)
    assert segments.tolist() == [[1, 3], [5, 6], [9, 11]]
    assert base.merge_segments(segments, max_gap=2).tolist() == [[1, 6], [9, 11]]
    assert base.merge_segments(np.empty((0, 2), dtype=int), 2).shape == (0, 2)


def test_moving_rms_and_envelope_of_a_sine() -> None:
    t = np.arange(int(FS)) / FS
    x = 3.0 * np.sin(2 * np.pi * 300 * t)
    assert filters.moving_rms(x, FS, 0.01)[500:1500].mean() == pytest.approx(
        3 / np.sqrt(2), rel=0.02
    )
    assert filters.envelope(x)[200:-200].mean() == pytest.approx(3.0, rel=0.01)


def test_clear_fast_ripple_is_detected_at_the_right_time_and_frequency() -> None:
    x = background(20.0, 1)
    truth = add_burst(x, 10.0, 320.0, 8, snr_db=15, band=BANDS["fast_ripple"])
    det = detect_energy(x, FS, BANDS)
    fr = det[det.band == "fast_ripple"]
    assert evaluate.count_matches(fr, truth) == {"tp": 1, "fp": 0, "fn": 0}
    assert fr.iloc[0].freq_hz == pytest.approx(320.0, rel=0.15)


def test_ripple_is_found_in_the_ripple_band_only() -> None:
    x = background(20.0, 2)
    truth = add_burst(x, 8.0, 150.0, 8, snr_db=15, band=BANDS["ripple"])
    det = detect_energy(x, FS, BANDS)
    assert evaluate.count_matches(det[det.band == "ripple"], truth)["tp"] == 1
    assert det[det.band == "fast_ripple"].empty


def test_no_detection_on_background_or_within_edge_margin() -> None:
    x = background(120.0, 3)
    assert detect_energy(x, FS, BANDS).empty
    add_burst(x, 0.5, 320.0, 8, snr_db=20, band=BANDS["fast_ripple"])  # inside the 1 s margin
    assert detect_energy(x, FS, BANDS, edge_margin_s=1.0).empty


def test_energy_detector_is_fooled_by_sharp_spikes() -> None:
    """Documented weakness: an isolated spike (no HFO) yields a ripple-band detection."""
    p = simulate.SimulationParams(duration_s=60.0).replace(
        events={**NO_EVENTS, "rate_spike_per_min": 12.0}
    )
    sim = simulate.simulate_from_seed(p, 4)
    det = detect_energy(sim.signal_uv, FS, BANDS)
    spikes = sim.events.assign(
        onset_s=sim.events.onset_s - 0.01, offset_s=sim.events.offset_s + 0.01
    )
    hits = evaluate.count_matches(det[det.band == "ripple"], spikes)["tp"]
    assert hits >= 0.5 * len(spikes)


def test_stricter_threshold_shrinks_the_detected_time() -> None:
    """A higher threshold can split one segment in two (more detections), but every
    high-threshold segment lies inside a low-threshold one, so the detected time shrinks."""
    sim = simulate.simulate_from_seed(simulate.SimulationParams(duration_s=60.0), 5)
    n = len(sim.signal_uv)

    def covered(k: float) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        for row in detect_energy(
            sim.signal_uv, FS, BANDS, EnergyParams(threshold_k=k)
        ).itertuples():
            mask[int(row.onset_s * FS) : int(row.offset_s * FS)] = True
        return mask

    low, mid, high = covered(3.0), covered(5.0), covered(8.0)
    assert not np.any(mid & ~low) and not np.any(high & ~mid)


def test_detect_channels_adds_channel_column() -> None:
    data = np.vstack([background(20.0, 6), background(20.0, 7)])
    add_burst(data[1], 10.0, 320.0, 8, snr_db=15, band=BANDS["fast_ripple"])
    det = base.detect_channels(detect_energy, data, FS, ["A", "B"], bands=BANDS)
    assert det["channel"].tolist() == ["B"]
