"""Calibrate the simulator background on the real recordings.

For every analysed channel (excluded recordings skipped), the median-Welch PSD is fitted with
``S(f) = c f^-beta + floor`` on ``[calibration] fit_range_hz``. The medians of beta and of the
crossover frequency, over channels whose floor is reached inside the fit range, are the values
to copy into ``[simulation.background]`` of ``configs/default.toml``.

Usage:
    python scripts/03_calibrate_background.py [--config configs/default.toml]

Outputs:
    results/background_fit.csv           one row per channel
    results/background_fit_summary.csv   medians and fractions quoted in the docs
    figures/sim_background_calibration.png   real vs simulated spectra, fitted parameters
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne_bids
import numpy as np
import pandas as pd

from epidetect import io, plotting, qc, simulate
from epidetect.config import load_config
from epidetect.simulate import SimulationParams

NORMALISATION_BAND_HZ = (20.0, 40.0)  # spectra are compared in shape: each divided by its mean here
SIMULATED_CHANNELS = 20  # number of simulated backgrounds averaged in the figure


def fit_real_channels(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit every analysed channel; also return frequencies and normalised PSDs."""
    bids_root, task = cfg["paths"]["bids_root"], cfg["bids"]["task"]
    excluded = {tuple(x) for x in cfg["data"]["excluded_recordings"]}
    cal = cfg["calibration"]
    rows, spectra, freqs = [], [], None
    for patient in sorted(mne_bids.get_entity_vals(bids_root, "subject")):
        for session in io.BIDS_SESSIONS:
            if (patient, session) in excluded:
                continue
            raw = io.read_bids_recording(bids_root, patient, session, task)
            freqs, psd = qc.welch_psd(raw.get_data() / io.MICROVOLT, raw.info["sfreq"])
            for channel, channel_psd in zip(raw.ch_names, psd, strict=True):
                fit = simulate.fit_background_model(
                    freqs,
                    channel_psd,
                    tuple(cal["fit_range_hz"]),
                    cfg["recording"]["line_freq_hz"],
                    cal["line_exclusion_hz"],
                )
                rows.append({"patient": patient, "session": session, "channel": channel, **fit})
                spectra.append(normalise(freqs, channel_psd))
    return pd.DataFrame(rows), freqs, np.array(spectra)


def normalise(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Divide a PSD by its mean in the normalisation band (compare shapes, not levels)."""
    mask = (freqs >= NORMALISATION_BAND_HZ[0]) & (freqs <= NORMALISATION_BAND_HZ[1])
    return psd / psd[..., mask].mean(axis=-1, keepdims=True)


def summarise(fits: pd.DataFrame, fit_high_hz: float) -> pd.DataFrame:
    """Medians quoted in the docs; 'floor reached' = crossover inside the fit range."""
    reached = fits[fits["crossover_hz"] <= fit_high_hz]
    values = {
        "n_channels": len(fits),
        "fraction_floor_reached": len(reached) / len(fits),
        "median_beta_floor_reached": reached["beta"].median(),
        "median_crossover_hz_floor_reached": reached["crossover_hz"].median(),
        "q25_crossover_hz_floor_reached": reached["crossover_hz"].quantile(0.25),
        "q75_crossover_hz_floor_reached": reached["crossover_hz"].quantile(0.75),
        "median_beta_all": fits["beta"].median(),
        "median_rmse_log10": fits["rmse_log10"].median(),
    }
    return pd.DataFrame({"quantity": list(values), "value": list(values.values())})


def simulated_spectra(params: SimulationParams, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised PSDs of ``n`` simulated backgrounds (no events)."""
    rng = np.random.default_rng(seed)
    n_samples = int(params.duration_s * params.fs_hz)
    x = np.array(
        [simulate.colored_noise(n_samples, params.fs_hz, params.background, rng) for _ in range(n)]
    )
    freqs, psd = qc.welch_psd(x, params.fs_hz)
    return freqs, normalise(freqs, psd)


def plot(
    freqs: np.ndarray,
    real: np.ndarray,
    fits: pd.DataFrame,
    sim: tuple[np.ndarray, np.ndarray],
    params: SimulationParams,
    fit_range: tuple[float, float],
    path: Path,
) -> None:
    """Left: real PSD percentiles vs simulated median. Right: fitted beta and crossover."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax = axes[0]
    keep = freqs >= 1.0
    q10, q50, q90 = np.percentile(real[:, keep], [10, 50, 90], axis=0)
    ax.axvspan(*fit_range, color=plotting.GRID, alpha=0.5, lw=0, label="fit range")
    ax.fill_between(
        freqs[keep],
        q10,
        q90,
        color=plotting.NEUTRAL,
        alpha=0.35,
        lw=0,
        label="real channels, 10-90 %",
    )
    ax.loglog(freqs[keep], q50, color=plotting.INK, lw=1.2, label="real channels, median")
    sim_freqs, sim_psd = sim
    sim_keep = sim_freqs >= 1.0
    ax.loglog(
        sim_freqs[sim_keep],
        np.median(sim_psd[:, sim_keep], axis=0),
        color=plotting.ACCENT,
        lw=1.5,
        ls="--",
        label="simulated background (default parameters)",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(
        f"PSD / mean PSD in {NORMALISATION_BAND_HZ[0]:.0f}-{NORMALISATION_BAND_HZ[1]:.0f} Hz"
    )
    ax.set_title(
        f"Spectral shape: beta = {params.background.beta} above {params.background.knee_hz:.0f} Hz"
        f" ({params.background.beta_low} below), crossover = "
        f"{params.background.crossover_hz:.0f} Hz",
        loc="left",
    )
    ax.legend(loc="lower left", fontsize=8)

    reached = fits["crossover_hz"] <= fit_range[1]
    axes[1].hist(fits.loc[reached, "beta"], bins=25, color=plotting.SERIES[0])
    axes[1].hist(fits.loc[~reached, "beta"], bins=25, color=plotting.NEUTRAL, alpha=0.6)
    axes[1].axvline(params.background.beta, color=plotting.ACCENT, ls="--")
    axes[1].set_xlabel("Fitted beta")
    axes[1].set_ylabel("Channels")
    axes[1].set_title("Exponent (20-500 Hz fit)", loc="left")
    axes[2].hist(
        fits.loc[reached, "crossover_hz"],
        bins=25,
        color=plotting.SERIES[0],
        label=f"floor reached < {fit_range[1]:.0f} Hz (n={int(reached.sum())})",
    )
    axes[2].axvline(params.background.crossover_hz, color=plotting.ACCENT, ls="--", label="default")
    axes[2].set_xlabel("Crossover frequency (Hz)")
    axes[2].set_title("Where 1/f meets the floor", loc="left")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[1].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=plotting.SERIES[0], label="floor reached"),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=plotting.NEUTRAL,
                alpha=0.6,
                label=f"no floor (n={int((~reached).sum())})",
            ),
        ],
        fontsize=8,
        loc="upper left",
    )
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    results, figures = cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"]
    fit_range = tuple(cfg["calibration"]["fit_range_hz"])
    params = SimulationParams.from_config(cfg["simulation"])

    fits, freqs, real = fit_real_channels(cfg)
    fits.round(5).to_csv(results / "background_fit.csv", index=False, lineterminator="\n")
    summary = summarise(fits, fit_range[1])
    summary.round(4).to_csv(
        results / "background_fit_summary.csv", index=False, lineterminator="\n"
    )
    sim = simulated_spectra(params, cfg["simulation"]["seed"], SIMULATED_CHANNELS)
    plot(freqs, real, fits, sim, params, fit_range, figures / "sim_background_calibration.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
