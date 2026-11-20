"""Tests for epidetect.simulate: timing, SNR, spectrum, reproducibility of the ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from epidetect import filters, qc, simulate
from epidetect.simulate import SimulationParams

FS = 2000.0


@pytest.fixture
def params() -> SimulationParams:
    return SimulationParams(duration_s=60.0)


def events_only(sim: simulate.Simulation) -> np.ndarray:
    """Signal minus background: exactly the sum of the simulated events.

    (Shrinking the background instead would shrink the events too: their amplitude is set by
    an SNR relative to the background.)"""
    return sim.signal_uv - sim.background_uv


def test_same_seed_same_simulation_and_different_seed_differs(params: SimulationParams) -> None:
    a = simulate.simulate_from_seed(params, 1)
    b = simulate.simulate_from_seed(params, 1)
    c = simulate.simulate_from_seed(params, 2)
    np.testing.assert_array_equal(a.signal_uv, b.signal_uv)
    assert a.events.equals(b.events)
    assert not np.array_equal(a.signal_uv, c.signal_uv)


def test_events_are_where_the_ground_truth_says(params: SimulationParams) -> None:
    sim = simulate.simulate_from_seed(params.replace(events={"p_hfo_on_spike": 0.0}), 3)
    x = events_only(sim)
    for row in sim.events.itertuples():
        start, stop = int(row.onset_s * FS), int(row.offset_s * FS) + 1
        if row.kind != "spike":
            # The HFO envelope peaks at the announced centre.
            seg = np.abs(x[start:stop])
            peak_time = (start + np.argmax(seg)) / FS
            assert abs(peak_time - row.center_s) < 1.0 / row.freq_hz
    # HFO energy is entirely inside the announced intervals; spikes have a slow wave after
    # their "offset", so check HFO-only simulations for the outside part.
    hfo_only = simulate.simulate_from_seed(
        params.replace(events={"rate_spike_per_min": 0.0, "p_hfo_on_spike": 0.0}), 3
    )
    y = events_only(hfo_only)
    mask = np.zeros(len(y), dtype=bool)
    for row in hfo_only.events.itertuples():
        mask[int(row.onset_s * FS) - 1 : int(row.offset_s * FS) + 2] = True
    assert np.abs(y[~mask]).max() < 1e-9 * np.abs(y).max()


def test_min_gap_and_edge_margin(params: SimulationParams) -> None:
    sim = simulate.simulate_from_seed(params, 4)
    centres = sim.events["center_s"].to_numpy()
    assert np.all(np.diff(centres) >= params.events.min_gap_s - 1 / FS)
    assert centres.min() >= params.edge_margin_s
    assert centres.max() <= params.duration_s - params.edge_margin_s


def test_event_rate_matches_on_average(params: SimulationParams) -> None:
    long = params.replace(duration_s=600.0)
    sim = simulate.simulate_from_seed(long, 5)
    expected = 18.0 * (600.0 - 2 * long.edge_margin_s) / 60  # 3 classes x 6 per minute
    assert len(sim.events) == pytest.approx(expected, rel=0.1)
    counts = sim.events["kind"].value_counts()
    assert set(counts.index) == set(simulate.EVENT_KINDS)


def test_hfo_frequency_and_band(params: SimulationParams) -> None:
    sim = simulate.simulate_from_seed(params.replace(events={"p_hfo_on_spike": 0.0}), 6)
    x = events_only(sim)
    hfo = sim.events[sim.events.kind != "spike"]
    for row in hfo.itertuples():
        band = (
            params.events.ripple_band_hz
            if row.kind == "ripple"
            else params.events.fast_ripple_band_hz
        )
        assert band[0] <= row.freq_hz <= band[1]
        seg = x[int(row.onset_s * FS) : int(row.offset_s * FS) + 1]
        spectrum = np.abs(np.fft.rfft(seg, n=2**16))
        peak = np.fft.rfftfreq(2**16, 1 / FS)[np.argmax(spectrum)]
        assert peak == pytest.approx(row.freq_hz, abs=0.25 * row.freq_hz / row.n_cycles)


@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 10.0])
def test_snr_is_defined_in_the_event_band(params: SimulationParams, snr_db: float) -> None:
    p = params.replace(events={"snr_db": snr_db, "p_hfo_on_spike": 0.0})
    sim = simulate.simulate_from_seed(p, 7)
    for row in sim.events[sim.events.kind != "spike"].itertuples():
        start, stop = int(row.onset_s * FS), int(row.offset_s * FS) + 1
        event_only = events_only(sim)[start:stop]
        measured = 10 * np.log10(np.mean(event_only**2) / sim.band_power_uv2[row.kind])
        assert measured == pytest.approx(snr_db, abs=0.3)


def test_background_band_power_matches_filtered_background(params: SimulationParams) -> None:
    sim = simulate.simulate_from_seed(params, 8)
    edge = int(params.edge_margin_s * FS)
    fr = filters.bandpass(sim.background_uv, FS, params.events.fast_ripple_band_hz)
    assert sim.band_power_uv2["fast_ripple"] == pytest.approx(fr[edge:-edge].var())
    assert sim.background_uv.std() == pytest.approx(params.background.std_uv)


def test_background_spectrum_recovered_by_fit() -> None:
    """Round trip: generate with known (beta, crossover), fit the model back."""
    bg = simulate.BackgroundParams(beta=3.0, crossover_hz=200.0, std_uv=100.0)
    x = bg.std_uv * simulate.colored_noise(int(120 * FS), FS, bg, np.random.default_rng(9))
    freqs, psd = qc.welch_psd(x[None, :], FS)
    fit = simulate.fit_background_model(freqs, psd[0])
    assert fit["beta"] == pytest.approx(3.0, abs=0.15)
    assert fit["crossover_hz"] == pytest.approx(200.0, rel=0.15)


def test_spike_waveform_shape() -> None:
    wave = simulate.spike_waveform(FS, -500.0, 0.003, 0.008, 0.3, 0.06, 0.03)
    centre = len(wave) // 2
    # Peak = cusp amplitude minus the slow wave's tail at t = 0: -500 * (1 - 0.3 exp(-2)).
    assert np.argmin(wave) == centre
    assert wave[centre] == pytest.approx(-500.0 * (1 - 0.3 * np.exp(-2.0)))
    assert abs(wave[0]) < 1e-3 and abs(wave[-1]) < 1e-3  # no truncation discontinuity
    assert wave[centre + int(0.06 * FS)] > 0  # slow wave of opposite polarity


def _median_spike_ringing(params: SimulationParams, apex_ms: tuple[float, float]) -> float:
    """Median of max |250-500 Hz| near isolated spikes, in units of band background std."""
    p = params.replace(
        events={
            "rate_ripple_per_min": 0.0,
            "rate_fast_ripple_per_min": 0.0,
            "rate_spike_per_min": 12.0,
        },
        spike={"apex_sigma_ms": apex_ms},
    )
    sim = simulate.simulate_from_seed(p, 10)
    filtered = filters.bandpass(sim.signal_uv, FS, p.events.fast_ripple_band_hz)
    bg_std = np.sqrt(sim.band_power_uv2["fast_ripple"])
    peaks = [
        np.abs(filtered[int((r.center_s - 0.02) * FS) : int((r.center_s + 0.02) * FS)]).max()
        for r in sim.events.itertuples()
    ]
    return float(np.median(peaks) / bg_std)


def test_pure_cusp_spikes_create_strong_false_hfos(params: SimulationParams) -> None:
    """Isolated spikes (no HFO) with a sharp apex produce large 250-500 Hz bursts."""
    assert _median_spike_ringing(params, (0.0, 0.0)) > 20


def test_rounded_apex_reduces_ringing(params: SimulationParams) -> None:
    sharp = _median_spike_ringing(params, (0.0, 0.0))
    calibrated = _median_spike_ringing(params, (1.1, 1.6))
    smooth = _median_spike_ringing(params, (3.0, 3.0))
    assert sharp > 5 * calibrated > 5 * 1.0
    assert calibrated > smooth


def test_impossible_rate_raises() -> None:
    with pytest.raises(ValueError, match="impossible"):
        simulate.sample_event_centres(
            np.random.default_rng(0), 60.0, 1.0, rate_per_s=3.0, min_gap_s=0.5
        )
