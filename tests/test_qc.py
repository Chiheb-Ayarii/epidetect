"""Tests for epidetect.qc on synthetic signals with known spectral properties."""

from __future__ import annotations

import numpy as np
import pytest

from epidetect import qc

FS = 2000.0
N = int(60 * FS)  # 60 s gives ~119 Welch windows: a stable median PSD


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def brown_noise(rng: np.random.Generator, n_channels: int) -> np.ndarray:
    """Cumulative sum of white noise: PSD ~ 1/f^2 (an ideal integrator, beta = 2)."""
    x = np.cumsum(rng.normal(0, 1, (n_channels, N)), axis=1)
    return x - x.mean(axis=1, keepdims=True)


def test_aperiodic_slope_recovers_known_exponents(rng: np.random.Generator) -> None:
    white = rng.normal(0, 10, (1, N))
    brown = brown_noise(rng, 1)
    params = qc.QCParams()
    for signal, beta in [(white, 0.0), (brown, 2.0)]:
        freqs, psd = qc.welch_psd(signal, FS)
        assert qc.aperiodic_slope(freqs, psd, params)[0] == pytest.approx(beta, abs=0.2)


def test_line_noise_detected_only_where_present(rng: np.random.Generator) -> None:
    t = np.arange(N) / FS
    noise = rng.normal(0, 10, (2, N))
    mains = 20 * np.sin(2 * np.pi * 50 * t) + 5 * np.sin(2 * np.pi * 150 * t)
    data = noise + np.vstack([mains, np.zeros(N)])
    freqs, psd = qc.welch_psd(data, FS)
    harmonics, ratio_db = qc.line_noise_db(freqs, psd, qc.QCParams())
    i50, i100, i150 = (list(harmonics).index(f) for f in (50, 100, 150))
    assert ratio_db[0, i50] > 20 and ratio_db[0, i150] > 10
    assert abs(ratio_db[0, i100]) < 3  # no 100 Hz component was added
    assert np.all(np.abs(ratio_db[1]) < 3)  # clean channel


def test_band_power_of_sine_matches_variance() -> None:
    t = np.arange(N) / FS
    amplitude = 10.0
    data = amplitude * np.sin(2 * np.pi * 300 * t)[None, :]
    freqs, psd = qc.welch_psd(data, FS, average="mean")
    # Parseval: the power of a sine is A^2 / 2, all of it inside 250-500 Hz.
    assert qc.band_power(freqs, psd, (250, 500))[0] == pytest.approx(amplitude**2 / 2, rel=0.05)


def test_median_psd_is_unbiased_for_gaussian_noise(rng: np.random.Generator) -> None:
    sigma = 10.0
    data = rng.normal(0, sigma, (1, N))
    freqs, psd = qc.welch_psd(data, FS)  # median average with SciPy's bias correction
    # White noise variance is spread over 0..fs/2: total power = sigma^2.
    assert qc.band_power(freqs, psd, (0, FS / 2))[0] == pytest.approx(sigma**2, rel=0.05)


def test_robust_z_resists_one_outlier() -> None:
    values = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 50.0])
    z = qc.robust_z(values)
    assert z[-1] > 100 and np.all(np.abs(z[:-1]) < 2)
    assert np.all(qc.robust_z(np.ones(5)) == 0)


def test_channel_qc_flags_flat_and_noisy_channels(rng: np.random.Generator) -> None:
    data = 20 * brown_noise(rng, 8) / 100 + rng.normal(0, 2, (8, N))
    data[2] = rng.normal(0, 0.1, N)  # disconnected: flat
    data[5] += rng.normal(0, 200, N)  # broadband amplifier noise
    names = [f"{i:02d}-{i + 1:02d}" for i in range(1, 9)]
    table, _, _ = qc.channel_qc(data, FS, names)
    assert table.loc[table.bad, "channel"].tolist() == [names[2], names[5]]
    assert "flat" in table.loc[2, "reasons"]


def test_relative_criteria_skipped_with_few_channels(rng: np.random.Generator) -> None:
    data = rng.normal(0, 10, (3, N))
    data[1] *= 100  # would be an outlier, but 3 channels are too few to judge
    table, _, _ = qc.channel_qc(data, FS, ["1-2", "2-3", "3-4"])
    assert not table.bad.any()
    assert table["z_broad"].isna().all()


def test_transient_bursts_do_not_inflate_median_psd(rng: np.random.Generator) -> None:
    """Median Welch ignores sparse FR-like bursts; a mean would not."""
    t_burst = np.arange(int(0.02 * FS)) / FS
    burst = 100 * np.sin(2 * np.pi * 300 * t_burst) * np.hanning(t_burst.size)
    clean = rng.normal(0, 5, (1, N))
    bursty = clean.copy()
    for start in rng.choice(N - burst.size, size=20, replace=False):  # 20 bursts in 60 s
        bursty[0, start : start + burst.size] += burst
    f, p_clean = qc.welch_psd(clean, FS)
    _, p_bursty = qc.welch_psd(bursty, FS)
    band = (250.0, 500.0)
    ratio = qc.band_power(f, p_bursty, band)[0] / qc.band_power(f, p_clean, band)[0]
    assert ratio < 1.5
