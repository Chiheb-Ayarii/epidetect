"""Features describing an HFO candidate, used by the machine-learning detector.

Each candidate is an interval ``[onset_s, offset_s]`` in a given band, with a ``peak_s``. The
features fall in three groups:

* **oscillation**: duration, frequency, number of cycles, band energy and envelope peak,
  relative to the robust background level of the band;
* **spike indicators**: power in a lower band (20-80 Hz), raw peak-to-peak amplitude,
  sharpness (second difference), line length;
* **time-frequency**: peak of the whitened Morlet map inside the candidate and its valley
  ratio (the spike-versus-oscillation test of the wavelet detector, as a continuous value).

All amplitudes are normalised by robust statistics of the *same channel* (median / MAD), so a
model trained on some channels or patients can be applied to others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from epidetect import filters
from epidetect.detectors.base import zero_crossing_frequency
from epidetect.detectors.wavelet import WaveletParams, iter_whitened_chunks, valley_ratio
from epidetect.qc import robust_std

FEATURE_NAMES = [
    "duration_ms",
    "freq_hz",
    "n_cycles",
    "envelope_peak_z",
    "energy_ratio_db",
    "low_band_ratio_db",
    "raw_amplitude_z",
    "sharpness_z",
    "line_length_ratio",
    "tf_peak_z",
    "tf_valley_ratio",
]
EPS = 1e-12  # avoids log(0) and division by zero on flat segments


@dataclass(frozen=True)
class FeatureParams:
    """Feature extraction parameters (defaults mirror ``[features]`` in the config)."""

    low_band_hz: tuple[float, float] = (20.0, 80.0)
    context_ms: float = 50.0
    filter_order: int = 4

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> FeatureParams:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()})


def _tf_features(
    x_uv: np.ndarray,
    fs: float,
    candidates: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    wavelet_params: WaveletParams,
    edge_margin_s: float,
) -> pd.DataFrame:
    """Max whitened z inside each candidate (in its band) and the valley ratio at that point."""
    freqs = wavelet_params.freqs()
    out = pd.DataFrame(np.nan, index=candidates.index, columns=["tf_peak_z", "tf_valley_ratio"])
    peak_index = (candidates["peak_s"].to_numpy() * fs).round().astype(int)
    for chunk in iter_whitened_chunks(x_uv, fs, wavelet_params, edge_margin_s):
        inside = (peak_index >= chunk.core_start) & (peak_index < chunk.core_stop)
        for row_pos in np.flatnonzero(inside):
            row = candidates.iloc[row_pos]
            low, high = bands[row["band"]]
            f_rows = np.flatnonzero((freqs >= low) & (freqs <= high))
            start = max(0, int(row["onset_s"] * fs) - chunk.seg_start)
            stop = min(chunk.z.shape[1], int(np.ceil(row["offset_s"] * fs)) - chunk.seg_start + 1)
            window = chunk.z[f_rows, start:stop]
            if window.size == 0:
                continue
            f_rel, t_rel = np.unravel_index(np.argmax(window), window.shape)
            f_index, t_index = f_rows[f_rel], start + t_rel
            out.iloc[row_pos] = [
                float(chunk.z[f_index, t_index]),
                valley_ratio(chunk.z[:, t_index], freqs, f_index, wavelet_params),
            ]
    return out


def extract_features(
    x_uv: np.ndarray,
    fs: float,
    candidates: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    params: FeatureParams | None = None,
    wavelet_params: WaveletParams | None = None,
    edge_margin_s: float = 1.0,
) -> pd.DataFrame:
    """One row of ``FEATURE_NAMES`` per candidate (same index as ``candidates``)."""
    p = params or FeatureParams()
    wp = wavelet_params or WaveletParams()
    if candidates.empty:
        return pd.DataFrame(columns=FEATURE_NAMES, index=candidates.index, dtype=float)
    edge = int(edge_margin_s * fs)
    core = slice(edge, len(x_uv) - edge)
    band_signals = {
        name: filters.bandpass(x_uv, fs, band, p.filter_order) for name, band in bands.items()
    }
    band_scale = {name: robust_std(y[core]) + EPS for name, y in band_signals.items()}
    band_envelopes = {name: filters.envelope(y) for name, y in band_signals.items()}
    low = filters.bandpass(x_uv, fs, p.low_band_hz, p.filter_order)
    low_scale = robust_std(low[core]) + EPS
    raw_scale = robust_std(x_uv[core]) + EPS
    d1 = np.abs(np.diff(x_uv, prepend=x_uv[0]))
    d2 = np.abs(np.diff(x_uv, n=2, prepend=x_uv[:2]))
    d1_baseline, d2_scale = np.median(d1[core]) + EPS, robust_std(d2[core]) + EPS
    half_context = int(p.context_ms / 1000 * fs)

    rows = []
    for row in candidates.itertuples():
        start, stop = (
            int(row.onset_s * fs),
            max(int(np.ceil(row.offset_s * fs)), int(row.onset_s * fs) + 1),
        )
        centre = int(round(row.peak_s * fs))
        ctx = slice(max(0, centre - half_context), centre + half_context + 1)
        y = band_signals[row.band][start:stop]
        duration = (stop - start) / fs
        freq = zero_crossing_frequency(y, fs)
        rows.append(
            {
                "duration_ms": duration * 1000,
                "freq_hz": freq,
                "n_cycles": duration * freq,
                "envelope_peak_z": band_envelopes[row.band][start:stop].max()
                / band_scale[row.band],
                "energy_ratio_db": 10 * np.log10(np.mean(y**2) / band_scale[row.band] ** 2 + EPS),
                "low_band_ratio_db": 10 * np.log10(np.mean(low[ctx] ** 2) / low_scale**2 + EPS),
                "raw_amplitude_z": np.ptp(x_uv[ctx]) / raw_scale,
                "sharpness_z": d2[ctx].max() / d2_scale,
                "line_length_ratio": d1[start:stop].mean() / d1_baseline,
            }
        )
    table = pd.DataFrame(rows, index=candidates.index)
    tf = _tf_features(x_uv, fs, candidates, bands, wp, edge_margin_s)
    return pd.concat([table, tf], axis=1)[FEATURE_NAMES]
