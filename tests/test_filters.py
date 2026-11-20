"""Tests for epidetect.filters, including the 'filtered spike looks like an HFO' effect."""

from __future__ import annotations

import numpy as np
import pytest

from epidetect import filters

FS = 2000.0
T = np.arange(-0.5, 0.5, 1 / FS)
FR_BAND = (250.0, 500.0)


def hann_burst(freq: float, n_cycles: float, amp: float = 1.0) -> np.ndarray:
    """Hann-windowed sine centred at t = 0."""
    duration = n_cycles / freq
    window = np.where(np.abs(T) <= duration / 2, np.cos(np.pi * T / duration) ** 2, 0.0)
    return amp * window * np.sin(2 * np.pi * freq * T)


def test_passband_gain_is_one_and_stopband_is_attenuated() -> None:
    for freq, expected in [(354.0, 1.0), (50.0, 0.0), (900.0, 0.0)]:
        x = np.sin(2 * np.pi * freq * T)
        y = filters.bandpass(x, FS, FR_BAND)
        core = slice(len(T) // 4, 3 * len(T) // 4)  # ignore edge transients
        gain = y[core].std() / x[core].std()
        assert gain == pytest.approx(expected, abs=0.01)


def test_zero_phase_keeps_event_timing() -> None:
    # Centred on the geometric band centre sqrt(250 * 500) ~ 354 Hz: near the cut-offs the
    # filtfilt gain drops to 0.5 (-3 dB per pass, squared), which would distort the burst.
    x = hann_burst(354.0, 8)
    y = filters.bandpass(x, FS, FR_BAND)
    # The envelope maximum of the filtered burst stays at t = 0 (no group delay).
    assert abs(T[np.argmax(np.abs(y))]) <= 2 / FS
    np.testing.assert_allclose(y, x, atol=0.05 * np.abs(x).max())


def test_invalid_band_raises() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        filters.bandpass(np.zeros(100), FS, (250.0, 1200.0))


def test_sharp_spike_rings_but_smooth_spike_does_not() -> None:
    """A cusp-shaped spike produces a 250-500 Hz 'oscillation' larger than a real 5 uV FR;
    a Gaussian spike of the same amplitude produces almost nothing (its spectrum has no
    energy at 250+ Hz). Sharpness, not amplitude, creates false HFOs."""
    cusp = -500.0 * np.where(T < 0, np.exp(T / 0.003), np.exp(-T / 0.008))
    smooth = -500.0 * np.exp(-0.5 * (T / 0.005) ** 2)
    true_fr = hann_burst(300.0, 6, amp=5.0)
    peak = {
        name: np.abs(filters.bandpass(x, FS, FR_BAND)).max()
        for name, x in [("cusp", cusp), ("smooth", smooth), ("fr", true_fr)]
    }
    assert peak["cusp"] > 3 * peak["fr"]
    assert peak["smooth"] < 0.01 * peak["fr"]


def test_narrower_band_rings_longer() -> None:
    impulse = np.zeros_like(T)
    impulse[len(T) // 2] = 1.0

    def ringing_ms(band: tuple[float, float]) -> float:
        y = np.abs(filters.bandpass(impulse, FS, band))
        above = np.flatnonzero(y > 0.1 * y.max())
        return (above[-1] - above[0]) / FS * 1000

    assert ringing_ms((280.0, 320.0)) > 3 * ringing_ms((250.0, 500.0))
