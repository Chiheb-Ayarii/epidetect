"""Hybrid simulation: synthetic fast ripples inserted into real background (Roehri et al. 2017).

The background, its spikes and its artefacts are real (pre-resection windows without any
marked fast ripple); only the inserted fast ripples are synthetic, so their ground truth is
exact. Their properties are calibrated on the reference markings of the **training patients
only**:

* rate (markings per channel-minute);
* share riding on a spike (a large transient within ``on_spike_window_s`` of the FR centre):
  such FRs are centred on a real transient of the window;
* (frequency, duration) pairs, drawn jointly from real markings;
* in-band SNR: for a marking, ``10 log10(P_marking / P_background - 1)``, where ``P_marking`` is
  the 250-500 Hz power during the marking and ``P_background`` the squared robust std of the
  band on that channel (the marking power contains the background, hence the ``- 1``).

Caveat: a "background without marked FR" can still contain unmarked real FRs (the reference
is incomplete); detecting them counts as a false positive during tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from epidetect import filters, simulate
from epidetect.detectors.base import zero_crossing_frequency
from epidetect.qc import robust_std
from epidetect.realdata import find_transients


@dataclass(frozen=True)
class HybridParams:
    """Hybrid simulation parameters (defaults mirror ``[hybrid]`` in the config)."""

    window_s: float = 30.0
    windows_per_patient: int = 6
    marking_margin_s: float = 1.0
    edge_margin_s: float = 1.0
    on_spike_window_s: float = 0.05
    on_spike_jitter_s: float = 0.01
    min_gap_s: float = 0.5
    min_snr_db: float = -20.0
    fr_band_hz: tuple[float, float] = (250.0, 500.0)
    transient_band_hz: tuple[float, float] = (3.0, 40.0)
    transient_z: float = 8.0
    transient_min_gap_s: float = 0.3
    filter_order: int = 4
    seed: int = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> HybridParams:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()})


@dataclass(frozen=True)
class HybridCalibration:
    """Fast-ripple statistics of a set of (training) patients."""

    rate_per_min: float
    p_on_spike: float
    freq_hz: np.ndarray
    duration_s: np.ndarray
    snr_db: np.ndarray

    @classmethod
    def from_statistics(cls, stats: pd.DataFrame, minutes: float) -> HybridCalibration:
        """Pool per-marking statistics (``marking_statistics``) of several patients."""
        valid = stats.dropna(subset=["freq_hz", "snr_db"])
        return cls(
            rate_per_min=len(stats) / minutes,
            p_on_spike=float(stats["on_spike"].mean()),
            freq_hz=valid["freq_hz"].to_numpy(float),
            duration_s=valid["duration_s"].to_numpy(float),
            snr_db=valid["snr_db"].to_numpy(float),
        )

    def summary(self) -> dict[str, float]:
        return {
            "rate_per_min": self.rate_per_min,
            "p_on_spike": self.p_on_spike,
            "median_freq_hz": float(np.median(self.freq_hz)),
            "median_duration_ms": float(np.median(self.duration_s) * 1000),
            "median_snr_db": float(np.median(self.snr_db)),
            "n_markings": len(self.freq_hz),
        }


def marking_statistics(
    x_uv: np.ndarray, fs: float, markings: pd.DataFrame, params: HybridParams
) -> pd.DataFrame:
    """Frequency, duration, in-band SNR and spike context of every marking of one channel."""
    y = filters.bandpass(x_uv, fs, params.fr_band_hz, params.filter_order)
    background_power = robust_std(y) ** 2
    transients = find_transients(
        x_uv, fs, params.transient_band_hz, params.transient_z, params.transient_min_gap_s
    )
    rows = []
    for m in markings.itertuples():
        seg = y[int(m.onset_s * fs) : int((m.onset_s + m.duration_s) * fs) + 1]
        ratio = np.mean(seg**2) / background_power - 1.0
        snr = 10 * np.log10(ratio) if ratio > 0 else -np.inf
        centre = m.onset_s + m.duration_s / 2
        near = (
            transients.size > 0 and np.min(np.abs(transients - centre)) <= params.on_spike_window_s
        )
        rows.append(
            {
                "freq_hz": zero_crossing_frequency(seg, fs),
                "duration_s": float(m.duration_s),
                "snr_db": max(snr, params.min_snr_db),
                "on_spike": bool(near),
            }
        )
    return pd.DataFrame(rows, columns=["freq_hz", "duration_s", "snr_db", "on_spike"])


def background_window_starts(
    n_samples: int,
    fs: float,
    marking_intervals: np.ndarray,
    params: HybridParams,
    rng: np.random.Generator,
    n_windows: int,
    max_tries: int = 1000,
) -> list[int]:
    """Start samples of windows that contain no marking (widened by ``marking_margin_s``)."""
    length = int(params.window_s * fs)
    if n_samples <= length:
        return []
    starts: list[int] = []
    for _ in range(max_tries):
        if len(starts) == n_windows:
            break
        start = int(rng.integers(0, n_samples - length))
        t0, t1 = (
            start / fs - params.marking_margin_s,
            (start + length) / fs + params.marking_margin_s,
        )
        clashes = (
            (marking_intervals[:, 0] < t1) & (marking_intervals[:, 1] > t0)
            if len(marking_intervals)
            else []
        )
        overlaps_window = any(abs(start - s) < length for s in starts)
        if not np.any(clashes) and not overlaps_window:
            starts.append(start)
    return starts


def insert_fast_ripples(
    x_uv: np.ndarray,
    fs: float,
    calibration: HybridCalibration,
    params: HybridParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Insert calibrated synthetic fast ripples into a real background window.

    Returns the hybrid signal and its ground truth (``kind``, ``on_spike``, ``onset_s``,
    ``offset_s``, ``center_s``, ``freq_hz``, ``n_cycles``, ``snr_db``).
    """
    signal = np.asarray(x_uv, dtype=float).copy()
    y = filters.bandpass(signal, fs, params.fr_band_hz, params.filter_order)
    background_power = robust_std(y) ** 2
    transients = find_transients(
        signal, fs, params.transient_band_hz, params.transient_z, params.transient_min_gap_s
    )
    duration = len(signal) / fs
    usable = (params.edge_margin_s, duration - params.edge_margin_s)
    n_events = rng.poisson(calibration.rate_per_min * (usable[1] - usable[0]) / 60)
    free_transients = list(
        rng.permutation(transients[(transients > usable[0] + 0.1) & (transients < usable[1] - 0.1)])
    )
    centres: list[float] = []
    rows = []
    for _ in range(n_events):
        j = int(rng.integers(len(calibration.freq_hz)))
        freq, length_s, snr = (
            calibration.freq_hz[j],
            calibration.duration_s[j],
            calibration.snr_db[j],
        )
        half = length_s / 2
        centre = None
        if rng.random() < calibration.p_on_spike:
            while free_transients and centre is None:
                t = free_transients.pop() + rng.uniform(
                    -params.on_spike_jitter_s, params.on_spike_jitter_s
                )
                if all(abs(t - c) >= params.min_gap_s for c in centres):
                    centre = t
        for _ in range(100):  # off-spike (or no transient left): uniform position
            if centre is not None:
                break
            t = rng.uniform(usable[0] + half, usable[1] - half)
            if all(abs(t - c) >= params.min_gap_s for c in centres):
                centre = t
        if centre is None:
            continue
        unit = simulate.hfo_waveform(fs, freq, length_s * freq, phase=rng.uniform(0, 2 * np.pi))
        amplitude = np.sqrt(10 ** (snr / 10) * background_power / np.mean(unit**2))
        index = int(round(centre * fs))
        start = index - len(unit) // 2
        signal[start : start + len(unit)] += amplitude * unit
        centres.append(index / fs)
        on_spike = (
            transients.size > 0
            and np.min(np.abs(transients - index / fs)) <= params.on_spike_window_s
        )
        rows.append(
            {
                "kind": "fast_ripple",
                "on_spike": bool(on_spike),
                "center_s": index / fs,
                "onset_s": index / fs - half,
                "offset_s": index / fs + half,
                "freq_hz": float(freq),
                "n_cycles": float(length_s * freq),
                "snr_db": float(snr),
            }
        )
    columns = [
        "kind",
        "on_spike",
        "center_s",
        "onset_s",
        "offset_s",
        "freq_hz",
        "n_cycles",
        "snr_db",
    ]
    truth = pd.DataFrame(rows, columns=columns).sort_values("center_s", ignore_index=True)
    return signal, truth
