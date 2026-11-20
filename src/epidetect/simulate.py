"""Simulator of single-channel intracranial signals with known ground truth.

Signal = background + events, sampled at ``fs_hz``:

* **Background**: Gaussian noise whose power spectrum is ``S(f) = 1/f^beta + floor``, with a
  shallower exponent ``beta_low`` below a knee (20 Hz). The 1/f part models brain background;
  the white floor models amplifier noise (visible in the real data above ~150-300 Hz, see
  docs/DATA_NOTES.md). The floor is set by the *crossover frequency* at which both terms are
  equal. Parameters are calibrated on the real channels (scripts/03_calibrate_background.py).
* **Ripples** (80-250 Hz) and **fast ripples** (250-500 Hz): Hann-windowed sinusoids with
  4-10 cycles, random frequency within the band and random phase.
* **Spikes**: cusp-shaped transients (exponential rise and decay) followed by a smooth slow
  wave. The cusp gives them a 1/f^2 spectrum, so a band-pass filter turns them into short
  "false HFOs". Spikes appear alone or under an HFO (``on_spike``).

Event amplitudes are set by a signal-to-noise ratio defined *in the event's band*:
``SNR_dB = 10 log10(mean power of the event over its support / power of the band-passed
background)``.

Event centres follow a renewal process: inter-event interval = ``min_gap_s`` + exponential.
Events therefore never overlap, which keeps the matching with detections unambiguous, and the
mean rate is exactly the requested one.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from epidetect import filters

EVENT_KINDS = ("ripple", "fast_ripple", "spike")
HFO_KINDS = ("ripple", "fast_ripple")
SPIKE_EXTENT_FRACTION = 0.05  # spike onset/offset: where the cusp falls below 5 % of its peak
SUPPORT_DECAY_CONSTANTS = 15.0  # exp(-15) ~ 3e-7: truncating there adds no spurious edges
SUPPORT_SIGMAS = 6.0  # Gaussian slow wave truncated at +/- 6 sigma


def _as_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


@dataclass(frozen=True)
class BackgroundParams:
    """Background spectrum (see :func:`background_psd_shape`), scaled to ``std_uv``."""

    beta: float = 3.5
    crossover_hz: float = 260.0
    beta_low: float = 2.5
    knee_hz: float = 20.0
    f_min_hz: float = 1.0
    std_uv: float = 140.0


@dataclass(frozen=True)
class EventParams:
    """Event classes, rates and amplitudes."""

    ripple_band_hz: tuple[float, float] = (80.0, 250.0)
    fast_ripple_band_hz: tuple[float, float] = (250.0, 500.0)
    n_cycles: tuple[float, float] = (4.0, 10.0)
    rate_ripple_per_min: float = 6.0
    rate_fast_ripple_per_min: float = 6.0
    rate_spike_per_min: float = 6.0
    min_gap_s: float = 0.5
    p_hfo_on_spike: float = 0.3
    snr_db: float = 10.0
    snr_jitter_db: float = 0.0
    # Optional empirical frequencies (Hz): if non-empty, fast-ripple frequencies are drawn
    # from these values (with replacement) instead of uniformly over the band.
    fast_ripple_freqs_hz: tuple[float, ...] = ()


@dataclass(frozen=True)
class SpikeParams:
    """Ranges (uniform draws) of the spike shape; amplitude in units of background std."""

    amplitude_std: tuple[float, float] = (3.0, 8.0)
    tau_rise_ms: tuple[float, float] = (1.0, 4.0)
    tau_decay_ms: tuple[float, float] = (4.0, 12.0)
    slow_wave_ratio: tuple[float, float] = (0.2, 0.5)
    slow_wave_delay_ms: tuple[float, float] = (40.0, 80.0)
    slow_wave_sigma_ms: tuple[float, float] = (20.0, 40.0)
    apex_sigma_ms: tuple[float, float] = (1.1, 1.6)


@dataclass(frozen=True)
class SimulationParams:
    """All simulator parameters (defaults mirror ``[simulation]`` in configs/default.toml)."""

    fs_hz: float = 2000.0
    duration_s: float = 60.0
    edge_margin_s: float = 1.0
    filter_order: int = 4
    background: BackgroundParams = field(default_factory=BackgroundParams)
    events: EventParams = field(default_factory=EventParams)
    spike: SpikeParams = field(default_factory=SpikeParams)

    @classmethod
    def from_config(cls, sim_cfg: dict[str, Any]) -> SimulationParams:
        """Build from the ``[simulation]`` config section (TOML lists become tuples)."""
        scalars = {
            k: v for k, v in sim_cfg.items() if k not in ("background", "events", "spike", "seed")
        }

        def sub(klass: type, key: str) -> Any:
            return klass(**{k: _as_tuple(v) for k, v in sim_cfg.get(key, {}).items()})

        return cls(
            **scalars,
            background=sub(BackgroundParams, "background"),
            events=sub(EventParams, "events"),
            spike=sub(SpikeParams, "spike"),
        )

    def replace(self, **changes: Any) -> SimulationParams:
        """Copy with some fields changed; ``background``/``events``/``spike`` accept dicts."""
        for key in ("background", "events", "spike"):
            if isinstance(changes.get(key), dict):
                changes[key] = dataclasses.replace(getattr(self, key), **changes[key])
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class Simulation:
    """A simulated signal and its ground truth.

    Attributes:
        signal_uv: Background plus events (uV).
        background_uv: Background alone (uV), kept to verify SNRs.
        fs: Sampling frequency (Hz).
        events: Ground truth, one row per event. Columns: ``event_id, kind, on_spike,
            center_s, onset_s, offset_s, duration_s, freq_hz, n_cycles, amplitude_uv, snr_db,
            spike_amplitude_uv``. ``kind`` is ``ripple``, ``fast_ripple`` or ``spike``;
            ``spike`` rows are isolated spikes (false-HFO generators).
        band_power_uv2: Power of the band-passed background in each HFO band (uV^2).
        params: The parameters used.
    """

    signal_uv: np.ndarray
    background_uv: np.ndarray
    fs: float
    events: pd.DataFrame
    band_power_uv2: dict[str, float]
    params: SimulationParams


# --- Background -------------------------------------------------------------------------------


def background_psd_shape(freqs: np.ndarray, bg: BackgroundParams) -> np.ndarray:
    """Relative background PSD (arbitrary scale).

    Aperiodic part: ``f^-beta`` above ``knee_hz``, ``f^-beta_low`` below it (continuous at the
    knee), flat below ``f_min_hz``. Plus a white floor equal to the aperiodic part at
    ``crossover_hz`` (which must lie above the knee).
    """
    f = np.maximum(freqs, bg.f_min_hz)
    high = f ** (-bg.beta)
    low = bg.knee_hz ** (-bg.beta) * (f / bg.knee_hz) ** (-bg.beta_low)
    aperiodic = np.where(f >= bg.knee_hz, high, low)
    return aperiodic + bg.crossover_hz ** (-bg.beta)


def colored_noise(
    n_samples: int, fs: float, bg: BackgroundParams, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian noise with the background spectrum, zero mean and unit standard deviation.

    Spectral shaping: FFT of white noise, multiplied by the *amplitude* spectrum
    ``sqrt(S(f))`` (power = amplitude squared), then inverse FFT.
    """
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    spectrum *= np.sqrt(background_psd_shape(freqs, bg))
    spectrum[0] = 0.0  # remove the DC component
    x = np.fft.irfft(spectrum, n=n_samples)
    return x / x.std()


def fit_background_model(
    freqs: np.ndarray,
    psd: np.ndarray,
    fit_range_hz: tuple[float, float] = (20.0, 500.0),
    line_freq_hz: float = 50.0,
    line_exclusion_hz: float = 3.0,
) -> dict[str, float]:
    """Fit ``log10 S(f) = log10(c f^-beta + floor)`` to one PSD by least squares in log space.

    Bins within ``line_exclusion_hz`` of a mains harmonic are ignored. The upper bound of the
    default range (500 Hz) stays below the anti-aliasing roll-off of the real recordings.

    Returns:
        ``beta``, ``crossover_hz`` (where ``c f^-beta = floor``), ``floor`` (uV^2/Hz) and
        ``rmse_log10`` (residual standard deviation, in decades).
    """
    mask = (freqs >= fit_range_hz[0]) & (freqs <= fit_range_hz[1])
    for harmonic in np.arange(line_freq_hz, fit_range_hz[1] + line_freq_hz, line_freq_hz):
        mask &= np.abs(freqs - harmonic) > line_exclusion_hz
    f, log_p = freqs[mask], np.log10(psd[mask])

    def residuals(theta: np.ndarray) -> np.ndarray:
        log_c, beta, log_floor = theta
        return np.log10(10.0**log_c * f ** (-beta) + 10.0**log_floor) - log_p

    start = np.array([log_p[0] + 2.5 * np.log10(f[0]), 2.5, log_p[-1]])
    result = least_squares(
        residuals, start, bounds=([-np.inf, 0.0, -np.inf], [np.inf, 6.0, np.inf])
    )
    log_c, beta, log_floor = result.x
    return {
        "beta": float(beta),
        "crossover_hz": float(10.0 ** ((log_c - log_floor) / beta)),
        "floor": float(10.0**log_floor),
        "rmse_log10": float(np.sqrt(np.mean(result.fun**2))),
    }


# --- Event waveforms --------------------------------------------------------------------------


def hfo_waveform(fs: float, freq_hz: float, n_cycles: float, phase: float = 0.0) -> np.ndarray:
    """Unit-amplitude Hann-windowed sinusoid lasting ``n_cycles / freq_hz`` seconds.

    The returned array is centred: its middle sample is the event centre.
    """
    duration = n_cycles / freq_hz
    half = int(round(duration * fs / 2))
    t = np.arange(-half, half + 1) / fs
    window = np.cos(np.pi * t / duration) ** 2  # Hann window, zero at +/- duration / 2
    return window * np.sin(2 * np.pi * freq_hz * t + phase)


def spike_waveform(
    fs: float,
    amplitude_uv: float,
    tau_rise_s: float,
    tau_decay_s: float,
    slow_wave_ratio: float,
    slow_wave_delay_s: float,
    slow_wave_sigma_s: float,
    apex_sigma_s: float = 0.0,
) -> np.ndarray:
    """Cusp-shaped spike followed by a slow wave of opposite polarity (centred array).

    ``amplitude * cusp(t) - amplitude * ratio * exp(-(t - delay)^2 / (2 sigma^2))`` where
    ``cusp`` rises as ``exp(t / tau_rise)`` and decays as ``exp(-t / tau_decay)``.

    The kink of a pure cusp at t = 0 gives a 1/f^2 spectrum, i.e. energy at hundreds of Hz.
    Real spikes have a rounded apex: ``apex_sigma_s > 0`` convolves the cusp with a Gaussian of
    that width, which multiplies its spectrum by ``exp(-(2 pi f sigma)^2 / 2)``. The cusp is then
    rescaled so that its peak is still ``amplitude_uv``. This single parameter controls how
    strongly a spike "rings" after band-pass filtering (calibrated on real data).
    """
    left = SUPPORT_DECAY_CONSTANTS * tau_rise_s + SUPPORT_SIGMAS * apex_sigma_s
    right = max(
        SUPPORT_DECAY_CONSTANTS * tau_decay_s + SUPPORT_SIGMAS * apex_sigma_s,
        slow_wave_delay_s + SUPPORT_SIGMAS * slow_wave_sigma_s,
    )
    half = int(np.ceil(max(left, right) * fs))  # symmetric support so the centre is the middle
    t = np.arange(-half, half + 1) / fs
    cusp = np.where(
        t < 0, np.exp(np.minimum(t, 0) / tau_rise_s), np.exp(-np.maximum(t, 0) / tau_decay_s)
    )
    if apex_sigma_s > 0:
        k = np.arange(-int(np.ceil(SUPPORT_SIGMAS * apex_sigma_s * fs)), 0)
        k = np.concatenate([k, [0], -k[::-1]]) / fs
        kernel = np.exp(-0.5 * (k / apex_sigma_s) ** 2)
        cusp = np.convolve(cusp, kernel / kernel.sum(), mode="same")
        cusp /= np.abs(cusp).max()
    slow = np.exp(-0.5 * ((t - slow_wave_delay_s) / slow_wave_sigma_s) ** 2)
    return amplitude_uv * (cusp - slow_wave_ratio * slow)


def _draw_spike(
    rng: np.random.Generator, sp: SpikeParams, fs: float, std_uv: float
) -> tuple[np.ndarray, float, float, float]:
    """Random spike: returns waveform, signed amplitude, rise and decay time constants (s)."""
    amplitude = rng.uniform(*sp.amplitude_std) * std_uv * rng.choice([-1.0, 1.0])
    tau_rise = rng.uniform(*sp.tau_rise_ms) / 1000
    tau_decay = rng.uniform(*sp.tau_decay_ms) / 1000
    wave = spike_waveform(
        fs,
        amplitude,
        tau_rise,
        tau_decay,
        rng.uniform(*sp.slow_wave_ratio),
        rng.uniform(*sp.slow_wave_delay_ms) / 1000,
        rng.uniform(*sp.slow_wave_sigma_ms) / 1000,
        apex_sigma_s=rng.uniform(*sp.apex_sigma_ms) / 1000,
    )
    return wave, amplitude, tau_rise, tau_decay


def _add_centred(signal: np.ndarray, wave: np.ndarray, centre_index: int) -> None:
    """Add a centred waveform into ``signal`` in place (it must fit entirely)."""
    half = len(wave) // 2
    start = centre_index - half
    if start < 0 or start + len(wave) > len(signal):
        raise ValueError("Event does not fit inside the signal; increase edge_margin_s.")
    signal[start : start + len(wave)] += wave


# --- Event timing -----------------------------------------------------------------------------


def sample_event_centres(
    rng: np.random.Generator,
    duration_s: float,
    edge_margin_s: float,
    rate_per_s: float,
    min_gap_s: float,
) -> np.ndarray:
    """Renewal process: interval = ``min_gap_s`` + Exponential(mean ``1/rate - min_gap_s``).

    No centre lies within ``edge_margin_s`` of either end of the signal.
    """
    if rate_per_s <= 0:
        return np.empty(0)
    mean_extra = 1.0 / rate_per_s - min_gap_s
    if mean_extra <= 0:
        raise ValueError(
            f"Total rate {rate_per_s * 60:.1f}/min is impossible with min_gap_s={min_gap_s}."
        )
    centres, t = [], edge_margin_s
    while True:
        t += min_gap_s + rng.exponential(mean_extra)
        if t > duration_s - edge_margin_s:
            return np.array(centres)
        centres.append(t)


# --- Simulation -------------------------------------------------------------------------------


def _band_power(background: np.ndarray, p: SimulationParams, band: tuple[float, float]) -> float:
    """Variance of the band-passed background, edges excluded (uV^2)."""
    edge = int(p.edge_margin_s * p.fs_hz)
    filtered = filters.bandpass(background, p.fs_hz, band, p.filter_order)
    return float(filtered[edge:-edge].var())


def simulate(params: SimulationParams, rng: np.random.Generator) -> Simulation:
    """Simulate one channel and its ground truth.

    Args:
        params: Simulator parameters.
        rng: Random generator; the same seed always gives the same simulation.
    """
    p, ev, bg = params, params.events, params.background
    n_samples = int(round(p.duration_s * p.fs_hz))
    background = bg.std_uv * colored_noise(n_samples, p.fs_hz, bg, rng)
    bands = {"ripple": ev.ripple_band_hz, "fast_ripple": ev.fast_ripple_band_hz}
    band_power = {kind: _band_power(background, p, band) for kind, band in bands.items()}

    rates = np.array([ev.rate_ripple_per_min, ev.rate_fast_ripple_per_min, ev.rate_spike_per_min])
    centres = sample_event_centres(
        rng, p.duration_s, p.edge_margin_s, rates.sum() / 60, ev.min_gap_s
    )
    # With all rates at 0, sample_event_centres returns no centre: background only.
    probabilities = rates / rates.sum() if rates.sum() > 0 else np.full(len(rates), 1 / len(rates))
    kinds = rng.choice(EVENT_KINDS, size=len(centres), p=probabilities)

    signal = background.copy()
    rows = []
    for event_id, (centre_s, kind) in enumerate(zip(centres, kinds, strict=True)):
        centre_index = int(round(centre_s * p.fs_hz))
        row: dict[str, Any] = {
            "event_id": event_id,
            "kind": kind,
            "center_s": centre_index / p.fs_hz,
        }
        if kind in HFO_KINDS:
            if kind == "fast_ripple" and ev.fast_ripple_freqs_hz:
                freq = float(rng.choice(ev.fast_ripple_freqs_hz))
            else:
                freq = rng.uniform(*bands[kind])
            n_cycles = rng.uniform(*ev.n_cycles)
            unit = hfo_waveform(p.fs_hz, freq, n_cycles, phase=rng.uniform(0, 2 * np.pi))
            snr_db = ev.snr_db + ev.snr_jitter_db * rng.standard_normal()
            target_power = 10.0 ** (snr_db / 10.0) * band_power[kind]
            amplitude = np.sqrt(target_power / np.mean(unit**2))
            _add_centred(signal, amplitude * unit, centre_index)
            duration = n_cycles / freq
            on_spike = bool(rng.random() < ev.p_hfo_on_spike)
            spike_amplitude = np.nan
            if on_spike:
                wave, spike_amplitude, _, _ = _draw_spike(rng, p.spike, p.fs_hz, bg.std_uv)
                _add_centred(signal, wave, centre_index)
            row |= {
                "on_spike": on_spike,
                "onset_s": row["center_s"] - duration / 2,
                "offset_s": row["center_s"] + duration / 2,
                "freq_hz": freq,
                "n_cycles": n_cycles,
                "amplitude_uv": amplitude,
                "snr_db": snr_db,
                "spike_amplitude_uv": spike_amplitude,
            }
        else:
            wave, spike_amplitude, tau_rise, tau_decay = _draw_spike(
                rng, p.spike, p.fs_hz, bg.std_uv
            )
            _add_centred(signal, wave, centre_index)
            extent = -np.log(SPIKE_EXTENT_FRACTION)  # time constants until 5 % of the peak
            row |= {
                "on_spike": False,
                "onset_s": row["center_s"] - extent * tau_rise,
                "offset_s": row["center_s"] + extent * tau_decay,
                "freq_hz": np.nan,
                "n_cycles": np.nan,
                "amplitude_uv": np.nan,
                "snr_db": np.nan,
                "spike_amplitude_uv": spike_amplitude,
            }
        rows.append(row)

    columns = [
        "event_id", "kind", "on_spike", "center_s", "onset_s", "offset_s", "duration_s",
        "freq_hz", "n_cycles", "amplitude_uv", "snr_db", "spike_amplitude_uv",
    ]  # fmt: skip
    events = pd.DataFrame(rows, columns=columns)
    events["duration_s"] = events["offset_s"] - events["onset_s"]
    return Simulation(signal, background, p.fs_hz, events, band_power, p)


def simulate_from_seed(params: SimulationParams, seed: int) -> Simulation:
    """Convenience wrapper: ``simulate(params, np.random.default_rng(seed))``."""
    return simulate(params, np.random.default_rng(seed))
