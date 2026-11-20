"""Zero-phase band-pass filtering.

All filters are applied forward and backward (``filtfilt``): the phases cancel, so events are
not shifted in time. That is essential when detections are compared with ground-truth onsets.
The magnitude response is squared, so the effective order is twice the design order, and the
gain at the cut-off frequencies is 0.5 (-6 dB) instead of 1/sqrt(2) (-3 dB). Events close to a
band edge are therefore attenuated.

Edge effects: near the start and end of a signal the filter output is transient. SciPy pads the
signal by odd reflection, which limits but does not remove this; callers should ignore the
first/last ``edge_margin_s`` seconds (see ``configs/default.toml``).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def butter_bandpass_sos(fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    """Butterworth band-pass in second-order sections.

    Args:
        fs: Sampling frequency (Hz).
        band: ``(low, high)`` cut-off frequencies (Hz), with ``0 < low < high < fs / 2``.
        order: Design order *per edge*; SciPy returns a band-pass of order ``2 * order``.

    Returns:
        SOS coefficients, shape ``(order, 6)``. Second-order sections avoid the numerical
        instability of high-order transfer-function coefficients.
    """
    low, high = band
    if not 0 < low < high < fs / 2:
        raise ValueError(f"Invalid band {band} for fs={fs} Hz (Nyquist = {fs / 2} Hz).")
    return butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis.

    Args:
        x: Signal(s), shape ``(..., n_samples)``.
        fs: Sampling frequency (Hz).
        band: ``(low, high)`` pass band (Hz).
        order: Design order per edge (effective order after filtfilt: ``4 * order``).
    """
    return sosfiltfilt(butter_bandpass_sos(fs, band, order), x, axis=-1)
