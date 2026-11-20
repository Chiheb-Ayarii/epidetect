"""RMS-energy HFO detector (in the spirit of Staba et al., J Neurophysiol 2002).

For each band (ripple 80-250 Hz, fast ripple 250-500 Hz):

1. zero-phase band-pass filter;
2. moving RMS over a short window (3 ms);
3. threshold = median(RMS) + k * robust_std(RMS): the median and the MAD are not inflated by
   the HFOs themselves, unlike the mean and the standard deviation;
4. supra-threshold segments closer than ``merge_gap_ms`` are merged, then segments shorter
   than ``min_duration_ms`` are dropped;
5. an *oscillation* criterion: at least ``min_peaks`` peaks of the rectified filtered signal
   above ``peak_threshold_k`` robust standard deviations of the filtered signal.

Step 5 is what separates an oscillation from a single large deflection, but as shown in step 2
a sharp spike still produces a few such peaks after filtering: this detector has no explicit
protection against filtered spikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from epidetect import filters
from epidetect.detectors.base import (
    empty_detections,
    mask_to_segments,
    merge_segments,
    zero_crossing_frequency,
)
from epidetect.qc import robust_std


@dataclass(frozen=True)
class EnergyParams:
    """Parameters of the RMS detector (defaults mirror ``[detectors.energy]`` in the config)."""

    filter_order: int = 4
    rms_window_ms: float = 3.0
    threshold_k: float = 5.0
    merge_gap_ms: float = 10.0
    min_duration_ms: float = 6.0
    min_peaks: int = 6
    peak_threshold_k: float = 3.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> EnergyParams:
        return cls(**cfg)


def detect_energy(
    x_uv: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    params: EnergyParams | None = None,
    edge_margin_s: float = 1.0,
) -> pd.DataFrame:
    """Detect HFOs in one channel, band by band.

    Args:
        x_uv: Signal (uV), 1-D.
        fs: Sampling frequency (Hz).
        bands: ``{"ripple": (80, 250), "fast_ripple": (250, 500)}`` or a subset.
        params: Detector parameters.
        edge_margin_s: Detections are only searched away from both ends (filter transients).

    Returns:
        Detection table (see ``epidetect.detectors.base``); ``score`` = max RMS / threshold.
    """
    p = params or EnergyParams()
    tables = [_detect_band(x_uv, fs, name, band, p, edge_margin_s) for name, band in bands.items()]
    tables = [t for t in tables if not t.empty]
    if not tables:
        return empty_detections()
    return pd.concat(tables, ignore_index=True).sort_values("onset_s", ignore_index=True)


def _detect_band(
    x_uv: np.ndarray,
    fs: float,
    band_name: str,
    band: tuple[float, float],
    p: EnergyParams,
    edge_margin_s: float,
) -> pd.DataFrame:
    y = filters.bandpass(x_uv, fs, band, p.filter_order)
    rms = filters.moving_rms(y, fs, p.rms_window_ms / 1000)
    edge = int(edge_margin_s * fs)
    core = slice(edge, len(y) - edge)
    threshold = np.median(rms[core]) + p.threshold_k * robust_std(rms[core])
    peak_height = p.peak_threshold_k * robust_std(y[core])

    mask = np.zeros(len(y), dtype=bool)
    mask[core] = rms[core] > threshold
    segments = merge_segments(mask_to_segments(mask), int(p.merge_gap_ms / 1000 * fs))
    rows = []
    for start, stop in segments:
        if (stop - start) / fs * 1000 < p.min_duration_ms:
            continue
        peaks, _ = find_peaks(np.abs(y[start:stop]), height=peak_height)
        if len(peaks) < p.min_peaks:
            continue
        peak_index = start + int(np.argmax(rms[start:stop]))
        rows.append(
            {
                "onset_s": start / fs,
                "offset_s": stop / fs,
                "peak_s": peak_index / fs,
                "band": band_name,
                "freq_hz": zero_crossing_frequency(y[start:stop], fs),
                "score": float(rms[peak_index] / threshold),
            }
        )
    return pd.DataFrame(rows, columns=empty_detections().columns)
