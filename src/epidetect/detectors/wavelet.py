"""Time-frequency HFO detector with Morlet wavelets.

Inspired by the principle published by Roehri et al. (IEEE TBME 2016) and implemented in their
Delphos software; this is an independent implementation, no code was copied.

1. **Morlet transform.** A complex Morlet wavelet with ``n_cycles`` cycles is, in the frequency
   domain, a Gaussian band-pass centred on f with width sigma_f = f / n_cycles (constant Q).
   We compute it with one FFT of the signal and one inverse FFT per frequency. The gain is
   normalised so that a sinusoid of amplitude A gives |W| = A.
2. **Whitening.** The 1/f background makes high frequencies invisible in a raw time-frequency
   map. Each frequency row is turned into a robust z-score over time (median, MAD), so the map
   becomes flat and any local excess stands out.
3. **Peaks.** Local maxima of the whitened map above ``z_threshold`` inside the HFO bands.
4. **Oscillation or spike?** At the peak time, a genuine oscillation is an isolated *blob*:
   just below its frequency (0.5-0.65 x f_peak) the z-score falls back to the background (a
   "valley"). A sharp spike is broadband: its z-score changes slowly with frequency and stays
   high there. A peak is accepted only if ``min z (valley region) < valley_ratio * z_peak``.

Long recordings are processed in chunks (with padding to avoid edge effects) so that memory
stays small.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from scipy.ndimage import maximum_filter

from epidetect.detectors.base import DETECTION_COLUMNS
from epidetect.qc import MAD_TO_STD


@dataclass(frozen=True)
class WaveletParams:
    """Parameters of the wavelet detector (defaults mirror ``[detectors.wavelet]``)."""

    freq_min_hz: float = 30.0
    freq_max_hz: float = 600.0
    n_freqs: int = 60
    n_cycles: float = 6.0
    chunk_s: float = 10.0
    pad_s: float = 0.5
    z_threshold: float = 8.0
    extent_ratio: float = 0.5
    valley_ratio: float = 0.5
    valley_low_fraction: float = 0.5
    valley_high_fraction: float = 0.65
    peak_neighbourhood_ms: float = 20.0
    peak_neighbourhood_freqs: int = 5

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> WaveletParams:
        return cls(**cfg)

    def freqs(self) -> np.ndarray:
        """Log-spaced analysis frequencies (constant Q: equal spacing in log f)."""
        return np.geomspace(self.freq_min_hz, self.freq_max_hz, self.n_freqs)


def morlet_transform(x: np.ndarray, fs: float, freqs: np.ndarray, n_cycles: float) -> np.ndarray:
    """Complex Morlet transform, shape ``(len(freqs), len(x))``, amplitude-normalised.

    In the frequency domain each wavelet is ``2 exp(-(f' - f)^2 / (2 sigma_f^2))`` for f' > 0
    (and 0 for f' < 0), with ``sigma_f = f / n_cycles``. Keeping only positive frequencies gives
    the analytic (complex) output directly; the factor 2 compensates the half of the energy of a
    real sinusoid that sits at negative frequencies, so ``|W| = A`` for ``A cos(2 pi f t)``.
    Its time-domain width is ``sigma_t = n_cycles / (2 pi f)``.
    """
    n = len(x)
    nfft = next_fast_len(n)
    spectrum = np.fft.fft(x, nfft)
    f_axis = np.fft.fftfreq(nfft, 1.0 / fs)
    positive = f_axis > 0
    out = np.empty((len(freqs), n), dtype=np.complex64)
    for k, f0 in enumerate(freqs):
        sigma_f = f0 / n_cycles
        gain = np.where(positive, 2.0 * np.exp(-0.5 * ((f_axis - f0) / sigma_f) ** 2), 0.0)
        out[k] = np.fft.ifft(spectrum * gain)[:n]
    return out


def whiten(amplitude: np.ndarray, reference: slice) -> tuple[np.ndarray, np.ndarray]:
    """Robust z-score of each frequency row, statistics taken over ``reference`` columns.

    Returns:
        ``z`` (whitened map, used for detection) and ``excess = amplitude - median`` (uV, not
        divided by the scale). Dividing by a scale that decreases with frequency (1/f
        background) tilts every event towards higher frequencies, so the frequency of an event
        is estimated on ``excess`` instead (see ``refine_peak_frequency``).
    """
    ref = amplitude[:, reference]
    median = np.median(ref, axis=1, keepdims=True)
    scale = MAD_TO_STD * np.median(np.abs(ref - median), axis=1, keepdims=True)
    excess = amplitude - median
    z = excess / np.where(scale > 0, scale, np.inf)
    return z.astype(np.float32), excess.astype(np.float32)


@dataclass(frozen=True)
class WhitenedChunk:
    """Whitened map of one chunk: columns cover samples ``[seg_start, seg_start + z.shape[1])``;
    only ``[core_start, core_stop)`` (absolute sample indices) belongs to this chunk."""

    seg_start: int
    core_start: int
    core_stop: int
    z: np.ndarray
    excess: np.ndarray


def iter_whitened_chunks(
    x: np.ndarray, fs: float, params: WaveletParams, edge_margin_s: float
) -> Iterator[WhitenedChunk]:
    """Whitened time-frequency maps of consecutive chunks (padded, edges excluded)."""
    n = len(x)
    edge, chunk, pad = (int(v * fs) for v in (edge_margin_s, params.chunk_s, params.pad_s))
    freqs = params.freqs()
    for core_start in range(edge, n - edge, chunk):
        core_stop = min(core_start + chunk, n - edge)
        seg_start, seg_stop = max(0, core_start - pad), min(n, core_stop + pad)
        amplitude = np.abs(morlet_transform(x[seg_start:seg_stop], fs, freqs, params.n_cycles))
        reference = slice(core_start - seg_start, core_stop - seg_start)
        z, excess = whiten(amplitude, reference)
        yield WhitenedChunk(seg_start, core_start, core_stop, z, excess)


def valley_ratio(
    z_column: np.ndarray, freqs: np.ndarray, f_index: int, params: WaveletParams
) -> float:
    """min z in [valley_low_fraction, valley_high_fraction] x f_peak, divided by z at the peak.

    A Morlet wavelet at frequency f has a Gaussian frequency response of width f / n_cycles.
    An oscillation at f_peak, seen from 0.5-0.65 f_peak (more than 4 widths away with 6
    cycles), contributes almost nothing: z falls back to the background level and the ratio is
    small. A spike is broadband: its whitened profile changes slowly with frequency, so z just
    below the peak stays a large fraction of the peak. (Searching much lower, e.g. down to
    40 Hz, does not work: after whitening, a spike's z-score *decreases* towards low
    frequencies, where the 1/f background is huge, and would look like a valley.)
    """
    f_peak = freqs[f_index]
    region = (freqs >= params.valley_low_fraction * f_peak) & (
        freqs <= params.valley_high_fraction * f_peak
    )
    if not region.any():
        return float("nan")
    return float(z_column[region].min() / z_column[f_index])


def refine_peak_frequency(
    column: np.ndarray, freqs: np.ndarray, f_index: int, search_bins: int = 3
) -> float:
    """Frequency of the maximum of ``column`` near ``f_index``, refined by a parabola.

    ``column`` is the non-whitened excess amplitude at the event time. The maximum is searched
    within ``search_bins`` of the whitened peak, then refined: the grid is log-spaced (~5 %
    steps with the defaults), too coarse to tell a 245 Hz ripple from a 255 Hz fast ripple.
    Fitting a parabola at (k-1, k, k+1) in log-frequency gives the sub-bin position
    delta = (c[k-1] - c[k+1]) / (2 (c[k-1] - 2 c[k] + c[k+1])).
    """
    low, high = max(0, f_index - search_bins), min(len(freqs), f_index + search_bins + 1)
    f_index = low + int(np.argmax(column[low:high]))
    if f_index == 0 or f_index == len(freqs) - 1:
        return float(freqs[f_index])
    left, centre, right = column[f_index - 1 : f_index + 2].astype(float)
    curvature = left - 2 * centre + right
    if curvature >= 0:  # not a strict maximum: keep the grid value
        return float(freqs[f_index])
    delta = 0.5 * (left - right) / curvature
    step = np.log(freqs[f_index + 1] / freqs[f_index])
    return float(freqs[f_index] * np.exp(np.clip(delta, -0.5, 0.5) * step))


def _extent(z_row: np.ndarray, t_index: int, level: float) -> tuple[int, int]:
    """Contiguous run around ``t_index`` where ``z_row > level`` (stop exclusive)."""
    above = z_row > level
    start = t_index
    while start > 0 and above[start - 1]:
        start -= 1
    stop = t_index + 1
    while stop < len(z_row) and above[stop]:
        stop += 1
    return start, stop


def detect_wavelet(
    x_uv: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    params: WaveletParams | None = None,
    edge_margin_s: float = 1.0,
    return_rejected: bool = False,
) -> pd.DataFrame:
    """Detect HFOs in one channel with the whitened Morlet map.

    Returns:
        Detection table (``score`` = peak z-score) with an extra ``valley_ratio`` column.
        With ``return_rejected=True``, spike-like peaks are also returned, with
        ``spike_like = True``.
    """
    p = params or WaveletParams()
    freqs = p.freqs()
    hfo_low = min(b[0] for b in bands.values())
    hfo_high = max(b[1] for b in bands.values())
    in_bands = (freqs >= hfo_low) & (freqs <= hfo_high)
    size = (p.peak_neighbourhood_freqs, max(1, int(p.peak_neighbourhood_ms / 1000 * fs)))
    candidates = []
    for chunk in iter_whitened_chunks(x_uv, fs, p, edge_margin_s):
        z = chunk.z
        is_peak = (z == maximum_filter(z, size=size, mode="nearest")) & (z > p.z_threshold)
        is_peak &= in_bands[:, None]
        is_peak[:, : chunk.core_start - chunk.seg_start] = False
        is_peak[:, chunk.core_stop - chunk.seg_start :] = False
        for f_index, t_index in np.argwhere(is_peak):
            z_peak = float(z[f_index, t_index])
            start, stop = _extent(z[f_index], t_index, p.extent_ratio * z_peak)
            ratio = valley_ratio(z[:, t_index], freqs, f_index, p)
            freq = refine_peak_frequency(chunk.excess[:, t_index], freqs, f_index)
            if not hfo_low <= freq <= hfo_high:
                continue
            candidates.append(
                {
                    "onset_s": (chunk.seg_start + start) / fs,
                    "offset_s": (chunk.seg_start + stop) / fs,
                    "peak_s": (chunk.seg_start + t_index) / fs,
                    "band": _band_of(freq, bands),
                    "freq_hz": freq,
                    "score": z_peak,
                    "valley_ratio": ratio,
                    "spike_like": not ratio < p.valley_ratio,
                }
            )
    table = _non_maximum_suppression(
        pd.DataFrame(candidates, columns=[*DETECTION_COLUMNS, "valley_ratio", "spike_like"])
    )
    if not return_rejected:
        table = table[~table["spike_like"].astype(bool)].drop(columns="spike_like")
    return table.sort_values("onset_s", ignore_index=True)


def _band_of(freq: float, bands: dict[str, tuple[float, float]]) -> str:
    """Name of the band containing ``freq`` (the lower band wins at a shared edge)."""
    for name, (low, high) in bands.items():
        if low <= freq <= high:
            return name
    raise ValueError(f"{freq} Hz is outside every band")


def _non_maximum_suppression(table: pd.DataFrame) -> pd.DataFrame:
    """Within each band, keep the strongest peak among overlapping intervals.

    One HFO produces several local maxima (neighbouring frequencies, chunks); the strongest one
    decides whether the event is an oscillation or a spike.
    """
    kept = []
    for _, group in table.groupby("band", sort=False):
        group = group.sort_values("score", ascending=False)
        intervals: list[tuple[float, float]] = []
        for row in group.itertuples():
            if any(row.onset_s < off and on < row.offset_s for on, off in intervals):
                continue
            intervals.append((row.onset_s, row.offset_s))
            kept.append(row.Index)
    return table.loc[kept] if kept else table.iloc[0:0]
