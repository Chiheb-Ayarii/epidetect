"""Why does a band-pass-filtered spike look like an HFO? A deterministic demonstration.

Signals (no noise, no randomness):
    * a cusp spike (pure kink at the apex) and the same spike with a rounded apex (Gaussian
      width = middle of the calibrated range, configs/default.toml);
    * a smooth Gaussian spike of the same amplitude;
    * a genuine fast ripple (Hann-windowed 300 Hz sinusoid).

They are filtered with zero-phase band-pass filters and the output is described by its peak,
and by the duration and number of cycles during which its envelope stays above 50 %, 10 % and
2 % of its maximum. The envelope is the modulus of the analytic signal (Hilbert transform,
explained in step 3).

Usage:
    python scripts/05_spike_filter_demo.py [--config configs/default.toml]

Outputs:
    results/false_hfo_demo.csv     every number quoted in the docs about this demonstration
    figures/false_hfo_demo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin, hilbert

from epidetect import filters, plotting, simulate
from epidetect.config import load_config

FS = 2000.0
T = np.arange(-0.5, 0.5, 1 / FS)  # 1 s, event at t = 0, far from the edges
SPIKE_AMPLITUDE_UV = -500.0
TAU_RISE_S, TAU_DECAY_S = 0.003, 0.008
GAUSS_SIGMA_S = 0.005
FR_FREQ_HZ, FR_CYCLES, FR_AMPLITUDE_UV = 300.0, 6, 5.0
FR_BAND = (250.0, 500.0)
NARROW_BANDS = [(250.0, 500.0), (250.0, 350.0), (280.0, 320.0), (295.0, 305.0)]
FIR_TAPS = 257  # long FIR: close to an ideal (brick-wall) band-pass
ENVELOPE_LEVELS = (0.5, 0.1, 0.02)


def centred(wave: np.ndarray) -> np.ndarray:
    """Place a centred waveform in the middle of the time axis T."""
    out = np.zeros_like(T)
    start = len(T) // 2 - len(wave) // 2
    out[start : start + len(wave)] = wave
    return out


def make_signals(apex_sigma_s: float) -> dict[str, np.ndarray]:
    """The four test signals of the demonstration."""

    def spike(apex: float) -> np.ndarray:
        return centred(
            simulate.spike_waveform(
                FS, SPIKE_AMPLITUDE_UV, TAU_RISE_S, TAU_DECAY_S, 0.0, 0.05, 0.03, apex_sigma_s=apex
            )
        )

    fr = FR_AMPLITUDE_UV * simulate.hfo_waveform(FS, FR_FREQ_HZ, FR_CYCLES)
    return {
        "cusp spike (sharp apex)": spike(0.0),
        f"rounded spike (apex {apex_sigma_s * 1000:.2f} ms)": spike(apex_sigma_s),
        "smooth Gaussian spike": SPIKE_AMPLITUDE_UV * np.exp(-0.5 * (T / GAUSS_SIGMA_S) ** 2),
        f"true fast ripple ({FR_FREQ_HZ:.0f} Hz, {FR_AMPLITUDE_UV:.0f} uV)": centred(fr),
    }


def describe(y: np.ndarray) -> dict[str, float]:
    """Peak, and duration / cycles while the envelope exceeds each level of its maximum."""
    envelope = np.abs(hilbert(y))
    out = {"peak_uv": float(envelope.max())}
    for level in ENVELOPE_LEVELS:
        above = np.flatnonzero(envelope > level * envelope.max())
        segment = y[above[0] : above[-1] + 1]
        zero_crossings = np.count_nonzero(np.diff(np.signbit(segment)))
        tag = f"{int(level * 100)}pct"
        out[f"duration_{tag}_ms"] = (above[-1] - above[0] + 1) / FS * 1000
        out[f"cycles_{tag}"] = zero_crossings / 2
    return out


def relative_spectrum_at(x: np.ndarray, freq_hz: float) -> float:
    """|X(f)| / max |X| at the frequency bin closest to freq_hz."""
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    return float(spectrum[np.argmin(np.abs(freqs - freq_hz))] / spectrum.max())


def fir_bandpass(x: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase long FIR band-pass (window method)."""
    return filtfilt(firwin(FIR_TAPS, band, pass_zero=False, fs=FS), [1.0], x)


def build_table(signals: dict[str, np.ndarray], order: int) -> pd.DataFrame:
    rows = []
    for name, x in signals.items():
        y = filters.bandpass(x, FS, FR_BAND, order)
        rows.append(
            {
                "signal": name,
                "filter": f"Butterworth N={order}",
                "band_hz": "250-500",
                "spectrum_300hz_rel": relative_spectrum_at(x, 300.0),
                **describe(y),
            }
        )
    cusp = next(iter(signals.values()))
    for band in NARROW_BANDS[1:]:
        y = filters.bandpass(cusp, FS, band, order)
        rows.append(
            {
                "signal": "cusp spike (sharp apex)",
                "filter": f"Butterworth N={order}",
                "band_hz": f"{band[0]:.0f}-{band[1]:.0f}",
                **describe(y),
            }
        )
    rows.append(
        {
            "signal": "cusp spike (sharp apex)",
            "filter": f"FIR {FIR_TAPS} taps",
            "band_hz": "250-500",
            **describe(fir_bandpass(cusp, FR_BAND)),
        }
    )
    return pd.DataFrame(rows)


def plot(signals: dict[str, np.ndarray], table: pd.DataFrame, order: int, path: Path) -> None:
    fig = plt.figure(figsize=(14, 8.5))
    grid = fig.add_gridspec(2, 6)
    ax_raw, ax_spec, ax_filt = (fig.add_subplot(grid[0, 2 * i : 2 * i + 2]) for i in range(3))
    ax_band, ax_gibbs = fig.add_subplot(grid[1, :3]), fig.add_subplot(grid[1, 3:])
    colors = [plotting.ACCENT, plotting.SERIES[0], plotting.NEUTRAL, plotting.INK]
    zoom = np.abs(T) <= 0.06
    t_ms = T * 1000

    for (name, x), color in zip(signals.items(), colors, strict=True):
        if "fast ripple" not in name:
            ax_raw.plot(t_ms[zoom], x[zoom], color=color, lw=1.2, label=name)
        spectrum = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), 1 / FS)
        keep = (freqs >= 5) & (freqs <= 1000)
        ax_spec.loglog(
            freqs[keep], spectrum[keep] / spectrum.max(), color=color, lw=1.2, label=name
        )
        y = filters.bandpass(x, FS, FR_BAND, order)
        ax_filt.plot(t_ms[zoom], y[zoom], color=color, lw=1.0, label=name)
    ax_raw.set_title("A. Raw transients (same -500 uV amplitude)", loc="left")
    ax_raw.set_xlabel("Time (ms)")
    ax_raw.set_ylabel("uV")
    ax_raw.legend(fontsize=7, loc="lower right")
    ax_spec.axvspan(*FR_BAND, color=plotting.BAND_FAST_RIPPLE, alpha=0.12, lw=0)
    ax_spec.set_ylim(1e-6, 2)
    ax_spec.set_title("B. Amplitude spectra (fast-ripple band shaded)", loc="left")
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("|X(f)| / max")
    ax_filt.set_title(
        f"C. After {FR_BAND[0]:.0f}-{FR_BAND[1]:.0f} Hz band-pass (zero phase)", loc="left"
    )
    ax_filt.set_xlabel("Time (ms)")
    ax_filt.set_ylabel("uV")
    ax_filt.legend(fontsize=7, loc="lower right")

    cusp = next(iter(signals.values()))
    zoom_wide = np.abs(T) <= 0.12
    for k, band in enumerate(NARROW_BANDS):
        y = filters.bandpass(cusp, FS, band, order)
        y = y / np.abs(y).max()
        row = table[
            (table.signal == "cusp spike (sharp apex)")
            & (table.band_hz == f"{band[0]:.0f}-{band[1]:.0f}")
            & table["filter"].str.startswith("Butter")
        ]
        ax_band.plot(t_ms[zoom_wide], y[zoom_wide] - 2.5 * k, color=plotting.INK, lw=0.7)
        ax_band.text(
            t_ms[zoom_wide][-1] + 3,
            -2.5 * k,
            f"{band[0]:.0f}-{band[1]:.0f} Hz: {row.cycles_50pct.iloc[0]:.0f} cycle(s) > 50 %",
            va="center",
            fontsize=8,
        )
    ax_band.set_yticks([])
    ax_band.set_xlim(t_ms[zoom_wide][0], t_ms[zoom_wide][-1] + 75)
    ax_band.set_xlabel("Time (ms)")
    ax_band.set_title(
        "D. The same cusp spike, narrower bands: longer 'oscillations' (each trace normalised)",
        loc="left",
    )

    for y, label, color in [
        (
            filters.bandpass(cusp, FS, FR_BAND, order),
            f"Butterworth N={order} (zero phase)",
            plotting.SERIES[0],
        ),
        (fir_bandpass(cusp, FR_BAND), f"FIR {FIR_TAPS} taps, near brick-wall", plotting.ACCENT),
    ]:
        envelope = np.abs(hilbert(y))
        ax_gibbs.semilogy(
            t_ms[zoom_wide], envelope[zoom_wide] / envelope.max(), color=color, lw=1.2, label=label
        )
    for level in ENVELOPE_LEVELS:
        ax_gibbs.axhline(level, color=plotting.GRID, lw=0.8, ls="--")
    ax_gibbs.set_ylim(1e-4, 1.5)
    ax_gibbs.set_xlabel("Time (ms)")
    ax_gibbs.set_ylabel("Envelope / max")
    ax_gibbs.set_title("E. Steeper filter, longer Gibbs tails (cusp spike, 250-500 Hz)", loc="left")
    ax_gibbs.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    apex_range = cfg["simulation"]["spike"]["apex_sigma_ms"]
    apex_sigma_s = float(np.mean(apex_range)) / 1000  # middle of the calibrated range
    order = cfg["simulation"]["filter_order"]
    signals = make_signals(apex_sigma_s)
    table = build_table(signals, order)
    table.to_csv(
        cfg["paths"]["results_dir"] / "false_hfo_demo.csv",
        index=False,
        float_format="%.4g",
        lineterminator="\n",
    )
    plot(signals, table, order, cfg["paths"]["figures_dir"] / "false_hfo_demo.png")
    columns = [
        "signal",
        "filter",
        "band_hz",
        "spectrum_300hz_rel",
        "peak_uv",
        "duration_50pct_ms",
        "cycles_50pct",
        "duration_10pct_ms",
        "cycles_10pct",
        "duration_2pct_ms",
        "cycles_2pct",
    ]
    print(table[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
