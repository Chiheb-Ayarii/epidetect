"""Signal quality control: per-channel spectrum, line noise, 1/f slope, bad-channel flags.

Design choices (see docs/PROJECT_WALKTHROUGH.md, step 1):

* The PSD is estimated with Welch's method using **median** averaging across 1 s windows. The
  median ignores short transients (fast ripples, artefacts) that occupy few windows, so the PSD
  describes the channel *background*. With mean averaging, an FR-rich channel would show
  inflated 250-500 Hz power and could be wrongly flagged as noisy.
* Channels are compared with the other channels of the same recording through a **robust**
  z-score (median / MAD), so that one aberrant channel cannot hide itself by inflating the
  standard deviation.
* Flagged channels are *marked*, never deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch

MAD_TO_STD = 1.4826  # MAD * 1.4826 = standard deviation for Gaussian data


@dataclass(frozen=True)
class QCParams:
    """Quality-control parameters (defaults mirror ``configs/default.toml``)."""

    welch_window_s: float = 1.0
    welch_overlap: float = 0.5
    broadband_hz: tuple[float, float] = (1.0, 500.0)
    ripple_band_hz: tuple[float, float] = (80.0, 250.0)
    fast_ripple_band_hz: tuple[float, float] = (250.0, 500.0)
    aperiodic_fit_hz: tuple[float, float] = (2.0, 40.0)
    line_freq_hz: float = 50.0
    line_half_width_hz: float = 1.0
    line_flank_inner_hz: float = 2.0
    line_flank_outer_hz: float = 5.0
    line_max_freq_hz: float = 450.0
    line_present_db: float = 6.0
    robust_z_threshold: float = 5.0
    min_channels_for_relative_qc: int = 4
    flat_std_uv: float = 1.0
    max_repeated_fraction: float = 0.1

    @classmethod
    def from_config(cls, qc_cfg: dict[str, Any], line_freq_hz: float) -> QCParams:
        """Build parameters from the ``[qc]`` config section (lists become tuples)."""
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in qc_cfg.items()}
        return cls(line_freq_hz=line_freq_hz, **values)


def welch_psd(
    data_uv: np.ndarray,
    fs: float,
    window_s: float = 1.0,
    overlap: float = 0.5,
    average: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with a Hann window (median-averaged by default).

    Note on the median: for Gaussian noise each periodogram bin is exponentially distributed
    and its median is ln(2) times its mean, so SciPy divides the median by ~ln(2). That
    correction is exact for random background but over-estimates a *deterministic* sine by
    ~1/ln(2) = 1.44. Use ``average="mean"`` to measure pure tones.

    Args:
        data_uv: Signals in uV, shape ``(n_channels, n_samples)``.
        fs: Sampling frequency (Hz).
        window_s: Window length (s); the frequency resolution is ``1 / window_s`` Hz.
        overlap: Fraction of overlap between consecutive windows.
        average: ``"median"`` (robust to transients) or ``"mean"``.

    Returns:
        ``freqs`` (Hz) and ``psd`` (uV^2/Hz, shape ``(n_channels, n_freqs)``).
    """
    nperseg = int(round(window_s * fs))
    return welch(
        data_uv,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=int(round(overlap * nperseg)),
        average=average,
        axis=-1,
    )


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Power (uV^2) in ``band`` by trapezoidal integration of the PSD."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def line_harmonics(line_freq_hz: float, max_freq_hz: float) -> np.ndarray:
    """Mains frequency and its harmonics up to ``max_freq_hz``: 50, 100, ..., 450 Hz."""
    return np.arange(line_freq_hz, max_freq_hz + 1e-9, line_freq_hz)


def line_noise_db(
    freqs: np.ndarray, psd: np.ndarray, params: QCParams
) -> tuple[np.ndarray, np.ndarray]:
    """Height of each mains harmonic above its spectral neighbourhood, in dB.

    For a harmonic ``h``: mean PSD within ``|f - h| <= half_width`` divided by the median PSD
    in the flanks ``inner <= |f - h| <= outer``. 0 dB means "no peak"; 10 dB means the peak is
    ten times the local background.

    Returns:
        ``harmonics`` (Hz) and ``ratio_db`` with shape ``(n_channels, n_harmonics)``.
    """
    harmonics = line_harmonics(params.line_freq_hz, params.line_max_freq_hz)
    ratio_db = np.empty((psd.shape[0], len(harmonics)))
    for k, h in enumerate(harmonics):
        distance = np.abs(freqs - h)
        peak = psd[:, distance <= params.line_half_width_hz].mean(axis=1)
        flank_mask = (distance >= params.line_flank_inner_hz) & (
            distance <= params.line_flank_outer_hz
        )
        flank = np.median(psd[:, flank_mask], axis=1)
        ratio_db[:, k] = 10.0 * np.log10(peak / flank)
    return harmonics, ratio_db


def aperiodic_slope(freqs: np.ndarray, psd: np.ndarray, params: QCParams) -> np.ndarray:
    """Exponent beta of a 1/f^beta fit (linear regression in log-log space).

    Bins within ``line_flank_outer_hz`` of a mains harmonic are excluded from the fit.
    A positive beta means the power decreases with frequency (beta ~ 2 for brown noise).
    """
    lo, hi = params.aperiodic_fit_hz
    mask = (freqs >= lo) & (freqs <= hi)
    for h in line_harmonics(params.line_freq_hz, hi + params.line_flank_outer_hz):
        mask &= np.abs(freqs - h) > params.line_flank_outer_hz
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[:, mask])
    slopes = np.polyfit(log_f, log_p.T, deg=1)[0]  # one fit per channel, vectorised
    return -slopes


def repeated_sample_fraction(data_uv: np.ndarray) -> np.ndarray:
    """Fraction of consecutive samples that are exactly equal, per channel.

    ~0 for a genuinely sampled analogue signal; 0.5 for a signal up-sampled by 2 by repeating
    every sample.
    """
    return (np.diff(data_uv, axis=-1) == 0).mean(axis=-1)


def robust_std(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """1.4826 * median absolute deviation: equals the std for Gaussian data, but a few large
    values (spikes, HFOs, artefacts) barely change it."""
    median = np.median(values, axis=axis, keepdims=axis is not None)
    mad = np.median(np.abs(values - median), axis=axis)
    return MAD_TO_STD * mad


def robust_z(values: np.ndarray) -> np.ndarray:
    """Robust z-score: (x - median) / (1.4826 * MAD). Returns zeros if the MAD is 0."""
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * MAD_TO_STD
    if mad == 0:
        return np.zeros_like(values, dtype=float)
    return (values - median) / mad


def channel_qc(
    data_uv: np.ndarray, fs: float, ch_names: list[str], params: QCParams | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-channel QC table for one recording.

    Columns: ``channel, std_uv, log10_power_broad, log10_power_ripple, log10_power_fr,
    aperiodic_beta, line_<f>_db`` for every harmonic, ``n_harmonics_present``, robust z-scores,
    ``bad`` and ``reasons``.

    A channel is flagged ``bad`` if (absolute criteria) its standard deviation is below
    ``flat_std_uv`` or more than ``max_repeated_fraction`` of its consecutive samples are
    identical, or (relative criteria, only with at least ``min_channels_for_relative_qc``
    channels) its broadband or fast-ripple background power is a robust-z outlier.

    Repeated samples: a channel recorded at fs/2 and up-sampled by repeating each sample
    (zero-order hold) has 50 % identical consecutive samples. Its effective Nyquist frequency
    is fs/4 (500 Hz here), so it cannot carry fast ripples although its file says 2000 Hz.
    Line noise is *reported* but not used to flag: it is handled later by filtering.

    Two PSDs are used: the median PSD (background, robust to FRs) for band powers and the 1/f
    slope, and the mean PSD for line noise, because mains harmonics are near-pure tones that
    the median bias correction would over-estimate by ~1.6 dB.

    Returns:
        The table, plus ``freqs`` and the median ``psd`` for plotting.
    """
    params = params or QCParams()
    win, ovl = params.welch_window_s, params.welch_overlap
    freqs, psd = welch_psd(data_uv, fs, win, ovl, average="median")
    _, psd_mean = welch_psd(data_uv, fs, win, ovl, average="mean")
    harmonics, ratio_db = line_noise_db(freqs, psd_mean, params)
    table = pd.DataFrame(
        {
            "channel": ch_names,
            "std_uv": data_uv.std(axis=1),
            "repeated_fraction": repeated_sample_fraction(data_uv),
            "log10_power_broad": np.log10(band_power(freqs, psd, params.broadband_hz)),
            "log10_power_ripple": np.log10(band_power(freqs, psd, params.ripple_band_hz)),
            "log10_power_fr": np.log10(band_power(freqs, psd, params.fast_ripple_band_hz)),
            "aperiodic_beta": aperiodic_slope(freqs, psd, params),
        }
    )
    for k, h in enumerate(harmonics):
        table[f"line_{int(h)}_db"] = ratio_db[:, k]
    table["n_harmonics_present"] = (ratio_db > params.line_present_db).sum(axis=1)

    relative = len(ch_names) >= params.min_channels_for_relative_qc
    table["z_broad"] = robust_z(table["log10_power_broad"].to_numpy()) if relative else np.nan
    table["z_fr"] = robust_z(table["log10_power_fr"].to_numpy()) if relative else np.nan

    reasons: list[str] = []
    for row in table.itertuples():
        why = []
        if row.std_uv < params.flat_std_uv:
            why.append("flat")
        if row.repeated_fraction > params.max_repeated_fraction:
            why.append(
                f"repeated samples ({row.repeated_fraction:.0%} identical consecutive samples: "
                "effective sampling rate below fs)"
            )
        if relative and abs(row.z_broad) > params.robust_z_threshold:
            why.append(f"broadband power outlier (z={row.z_broad:+.1f})")
        if relative and abs(row.z_fr) > params.robust_z_threshold:
            why.append(f"250-500 Hz background outlier (z={row.z_fr:+.1f})")
        reasons.append("; ".join(why))
    table["reasons"] = reasons
    table["bad"] = table["reasons"] != ""
    return table, freqs, psd
