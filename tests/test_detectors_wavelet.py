"""Tests for the Morlet time-frequency detector."""

from __future__ import annotations

import numpy as np
import pytest
from test_detectors_energy import BANDS, FS, NO_EVENTS, add_burst, background

from epidetect import evaluate, simulate
from epidetect.detectors import wavelet
from epidetect.detectors.energy import detect_energy
from epidetect.detectors.wavelet import WaveletParams, detect_wavelet


def test_morlet_amplitude_and_selectivity() -> None:
    t = np.arange(int(2 * FS)) / FS
    x = 4.0 * np.cos(2 * np.pi * 300 * t)
    w = np.abs(wavelet.morlet_transform(x, FS, np.array([300.0, 450.0]), n_cycles=6))
    core = slice(500, -500)
    assert w[0, core].mean() == pytest.approx(4.0, rel=0.01)  # |W| = A at the right frequency
    # The 450 Hz wavelet has sigma_f = 450 / 6 = 75 Hz: the 300 Hz sine is 2 sigma_f away,
    # so its response is A exp(-2^2 / 2) = 4 exp(-2) ~ 0.54.
    assert w[1, core].mean() == pytest.approx(4.0 * np.exp(-2.0), rel=0.01)


def test_whitening_gives_zero_median_and_unit_robust_scale() -> None:
    rng = np.random.default_rng(0)
    amplitude = np.abs(rng.normal(5.0, 2.0, (3, 5000)))
    z, excess = wavelet.whiten(amplitude, slice(None))
    assert np.allclose(np.median(z, axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(
        excess, amplitude - np.median(amplitude, axis=1, keepdims=True), rtol=1e-6
    )


def test_refine_peak_frequency_between_grid_points() -> None:
    freqs = np.geomspace(30, 600, 60)
    k = int(np.argmin(np.abs(freqs - 250)))
    true = np.sqrt(freqs[k] * freqs[k + 1])  # halfway between two bins in log scale
    column = -((np.log(freqs) - np.log(true)) ** 2)  # parabola in log f, maximum at `true`
    assert wavelet.refine_peak_frequency(column, freqs, k) == pytest.approx(true, rel=1e-6)


def test_clear_fast_ripple_detected_with_its_frequency() -> None:
    x = background(30.0, 1)
    truth = add_burst(x, 12.0, 320.0, 8, snr_db=12, band=BANDS["fast_ripple"])
    det = detect_wavelet(x, FS, BANDS)
    fr = det[det.band == "fast_ripple"]
    assert evaluate.count_matches(fr, truth) == {"tp": 1, "fp": 0, "fn": 0}
    assert fr.iloc[0].freq_hz == pytest.approx(320.0, rel=0.05)
    assert fr.iloc[0].valley_ratio < WaveletParams().valley_ratio


def test_burst_on_a_chunk_boundary_is_detected_once() -> None:
    x = background(30.0, 2)
    # Chunks start at the 1 s edge margin and last 10 s: 11 s is a boundary.
    truth = add_burst(x, 11.0, 300.0, 8, snr_db=12, band=BANDS["fast_ripple"])
    det = detect_wavelet(x, FS, BANDS)
    assert evaluate.count_matches(det, truth) == {"tp": 1, "fp": 0, "fn": 0}


def test_background_only_gives_no_detection() -> None:
    assert detect_wavelet(background(120.0, 3), FS, BANDS).empty


def test_sharp_spikes_are_rejected_but_fool_the_energy_detector() -> None:
    """Pure-cusp spikes are broadband: just below the time-frequency peak their whitened
    z-score stays high (no valley), so the wavelet detector rejects them."""
    p = simulate.SimulationParams(duration_s=60.0).replace(
        events={**NO_EVENTS, "rate_spike_per_min": 12.0}, spike={"apex_sigma_ms": (0.0, 0.0)}
    )
    sim = simulate.simulate_from_seed(p, 4)
    spikes = sim.events.assign(
        onset_s=sim.events.onset_s - 0.01, offset_s=sim.events.offset_s + 0.01
    )
    by_wavelet = evaluate.count_matches(detect_wavelet(sim.signal_uv, FS, BANDS), spikes)["tp"]
    by_energy = evaluate.count_matches(detect_energy(sim.signal_uv, FS, BANDS), spikes)["tp"]
    assert by_wavelet <= 0.2 * len(spikes) < by_energy
    rejected = detect_wavelet(sim.signal_uv, FS, BANDS, return_rejected=True)
    assert rejected["spike_like"].astype(bool).sum() >= 0.5 * len(spikes)


def test_hfo_riding_on_a_spike_is_rejected_documented_trade_off() -> None:
    """The cost of spike rejection: an HFO on a spike looks like a spike column with a bump."""
    p = simulate.SimulationParams(duration_s=60.0).replace(
        events={"rate_spike_per_min": 0.0, "p_hfo_on_spike": 1.0, "snr_db": 10.0}
    )
    sim = simulate.simulate_from_seed(p, 5)
    det = detect_wavelet(sim.signal_uv, FS, BANDS)
    assert evaluate.count_matches(det, sim.events)["tp"] <= 0.2 * len(sim.events)
