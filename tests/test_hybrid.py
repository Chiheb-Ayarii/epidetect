"""Tests for the hybrid simulation (synthetic fast ripples in a background)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epidetect import filters, hybrid, simulate
from epidetect.qc import robust_std

FS = 2000.0
PARAMS = hybrid.HybridParams()


def background_with_spikes(
    seed: int, duration_s: float = 30.0, n_spikes: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated background with a few sharp transients at known times (stands for real data)."""
    rng = np.random.default_rng(seed)
    bg = simulate.BackgroundParams()
    x = bg.std_uv * simulate.colored_noise(int(duration_s * FS), FS, bg, rng)
    times = np.linspace(3.0, duration_s - 3.0, n_spikes)
    spike = simulate.spike_waveform(FS, -1500.0, 0.003, 0.008, 0.3, 0.06, 0.03, apex_sigma_s=0.0012)
    for t in times:
        c = int(t * FS)
        x[c - len(spike) // 2 : c - len(spike) // 2 + len(spike)] += spike
    return x, times


def calibration(
    p_on_spike: float, rate: float = 20.0, snr: float = 10.0
) -> hybrid.HybridCalibration:
    return hybrid.HybridCalibration(
        rate_per_min=rate,
        p_on_spike=p_on_spike,
        freq_hz=np.array([300.0, 320.0]),
        duration_s=np.array([0.02, 0.025]),
        snr_db=np.array([snr, snr]),
    )


def test_marking_statistics_recovers_frequency_snr_and_context() -> None:
    x, spike_times = background_with_spikes(0)
    unit = simulate.hfo_waveform(FS, 300.0, 6)
    background_power = robust_std(filters.bandpass(x, FS, (250.0, 500.0))) ** 2
    amplitude = np.sqrt(10 ** (10 / 10) * background_power / np.mean(unit**2))
    rows = []
    for centre in (spike_times[1], 14.37):  # one on a spike, one away from spikes
        c = int(centre * FS)
        x[c - len(unit) // 2 : c - len(unit) // 2 + len(unit)] += amplitude * unit
        rows.append({"onset_s": centre - 0.01, "duration_s": 0.02})
    stats = hybrid.marking_statistics(x, FS, pd.DataFrame(rows), PARAMS)
    assert stats.on_spike.tolist() == [True, False]
    off = stats.iloc[1]
    assert off.freq_hz == pytest.approx(300.0, rel=0.1)
    assert off.snr_db == pytest.approx(10.0, abs=1.5)


def test_insertion_follows_the_calibration() -> None:
    x, spike_times = background_with_spikes(1)
    signal, truth = hybrid.insert_fast_ripples(
        x, FS, calibration(p_on_spike=1.0), PARAMS, np.random.default_rng(2)
    )
    assert len(truth) > 0
    # Every event goes on a real transient while some are left (6 here); extra events
    # are placed uniformly, as documented.
    assert truth.on_spike.sum() == min(len(truth), len(spike_times))
    assert np.all(np.diff(truth.center_s) >= PARAMS.min_gap_s - 1e-9)
    assert truth.center_s.min() >= PARAMS.edge_margin_s
    # Off-spike placement when p_on_spike = 0.
    _, truth0 = hybrid.insert_fast_ripples(
        x, FS, calibration(p_on_spike=0.0, rate=6.0), PARAMS, np.random.default_rng(3)
    )
    distance = np.abs(truth0.center_s.to_numpy()[:, None] - spike_times[None, :]).min(axis=1)
    assert truth0.on_spike.tolist() == list(distance <= PARAMS.on_spike_window_s)


def test_inserted_snr_matches_target() -> None:
    x, _ = background_with_spikes(4, n_spikes=0)
    signal, truth = hybrid.insert_fast_ripples(
        x, FS, calibration(0.0, rate=6.0, snr=5.0), PARAMS, np.random.default_rng(5)
    )
    background_power = robust_std(filters.bandpass(x, FS, (250.0, 500.0))) ** 2
    for row in truth.itertuples():
        seg = (signal - x)[int(row.onset_s * FS) : int(row.offset_s * FS) + 1]
        assert 10 * np.log10(np.mean(seg**2) / background_power) == pytest.approx(5.0, abs=0.5)


def test_background_windows_avoid_markings_and_each_other() -> None:
    markings = np.array([[10.0, 10.02], [50.0, 50.03]])
    starts = hybrid.background_window_starts(
        int(300 * FS), FS, markings, PARAMS, np.random.default_rng(0), 3
    )
    assert len(starts) == 3
    length = int(PARAMS.window_s * FS)
    for s in starts:
        t0, t1 = s / FS - PARAMS.marking_margin_s, (s + length) / FS + PARAMS.marking_margin_s
        assert not np.any((markings[:, 0] < t1) & (markings[:, 1] > t0))
    assert all(abs(a - b) >= length for i, a in enumerate(starts) for b in starts[i + 1 :])


def test_calibration_pools_statistics() -> None:
    stats = pd.DataFrame(
        {
            "freq_hz": [300.0, 280.0, np.nan],
            "duration_s": [0.02, 0.01, 0.02],
            "snr_db": [5.0, 2.0, 1.0],
            "on_spike": [True, False, True],
        }
    )
    cal = hybrid.HybridCalibration.from_statistics(stats, minutes=1.5)
    assert cal.rate_per_min == 2.0 and cal.p_on_spike == pytest.approx(2 / 3)
    assert len(cal.freq_hz) == 2 and cal.summary()["median_snr_db"] == 3.5
