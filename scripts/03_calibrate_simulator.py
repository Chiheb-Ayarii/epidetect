"""Calibrate the simulator on the real recordings: background spectrum and spike sharpness.

Both parts skip excluded recordings and QC-flagged channels (status "bad" in BIDS).

Part 1 - background. For every analysed channel, the median-Welch
PSD is fitted with ``S(f) = c f^-beta + floor`` on ``[calibration] fit_range_hz``. The medians
of beta and of the crossover frequency, over channels whose floor is reached inside the fit
range, are the values to copy into ``[simulation.background]`` of ``configs/default.toml``.

Part 2 - spikes. How much a spike "rings" after a 250-500 Hz band-pass depends on how rounded
its apex is. On real pre-resection data we find large transients (robust z > threshold in
3-40 Hz), away from marked FRs, and measure the ringing ratio = max |FR-band signal| within
+/- 30 ms / robust std of the FR-band signal. The same ratio is measured on simulated spikes for
a grid of apex-width ranges; the range whose quantiles best match the real ones (log scale) is
the value to copy into ``[simulation.spike] apex_sigma_ms``.

Usage:
    python scripts/03_calibrate_simulator.py [--config configs/default.toml]

Outputs:
    results/background_fit.csv            one row per channel
    results/background_fit_summary.csv    medians and fractions quoted in the docs
    results/spike_ringing_real.csv        one row per real transient
    results/spike_apex_grid.csv           simulated quantiles and distance to the real ones,
                                          for each candidate range
    results/spike_calibration_summary.csv real vs simulated quantiles of the ringing ratio
    figures/sim_background_calibration.png
    figures/sim_spike_calibration.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne_bids
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from epidetect import filters, io, plotting, qc, simulate
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
                if channel in raw.info["bads"]:  # QC-flagged in step 1 (channels.tsv)
                    continue
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


# --- Part 2: spike sharpness ------------------------------------------------------------------

RINGING_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def robust_std(x: np.ndarray) -> float:
    """1.4826 * median absolute deviation (std of Gaussian data, insensitive to bursts)."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def ringing_ratio(fr_band: np.ndarray, centre: int, half_window: int, scale: float) -> float:
    """max |FR-band signal| within +/- half_window samples of centre, divided by scale."""
    return float(np.abs(fr_band[centre - half_window : centre + half_window + 1]).max() / scale)


def measure_real_ringing(cfg: dict) -> pd.DataFrame:
    """Ringing ratio at large sharp transients of real pre-resection recordings."""
    cal = cfg["calibration"]
    bids_root, task = cfg["paths"]["bids_root"], cfg["bids"]["task"]
    excluded = {tuple(x) for x in cfg["data"]["excluded_recordings"]}
    order = cfg["simulation"]["filter_order"]
    fr_band = tuple(cfg["simulation"]["events"]["fast_ripple_band_hz"])
    rows = []
    for patient in sorted(mne_bids.get_entity_vals(bids_root, "subject")):
        if (patient, "pre") in excluded:
            continue
        raw = io.read_bids_recording(bids_root, patient, "pre", task)
        fs = raw.info["sfreq"]
        x = raw.get_data() / io.MICROVOLT
        low = filters.bandpass(x, fs, tuple(cal["transient_band_hz"]), order)
        high = filters.bandpass(x, fs, fr_band, order)
        markings = io.annotations_to_markings(raw)
        half = int(cal["ringing_half_window_s"] * fs)
        edge = int(cfg["simulation"]["edge_margin_s"] * fs)
        for i, channel in enumerate(raw.ch_names):
            if channel in raw.info["bads"]:  # QC-flagged in step 1 (channels.tsv)
                continue
            z = (low[i] - np.median(low[i])) / robust_std(low[i])
            peaks, _ = find_peaks(
                np.abs(z), height=cal["transient_z"], distance=int(cal["transient_min_gap_s"] * fs)
            )
            fr_times = markings.loc[markings.channel == channel, "onset_s"].to_numpy()
            scale = robust_std(high[i])
            for peak in peaks[(peaks > edge) & (peaks < x.shape[1] - edge)]:
                t = peak / fs
                if fr_times.size and np.min(np.abs(fr_times - t)) < cal["fr_exclusion_s"]:
                    continue  # a marked FR here would inflate the ratio: not a pure spike
                rows.append(
                    {
                        "patient": patient,
                        "channel": channel,
                        "time_s": t,
                        "z": abs(z[peak]),
                        "ratio": ringing_ratio(high[i], peak, half, scale),
                    }
                )
    return pd.DataFrame(rows)


def simulated_ringing(
    params: SimulationParams, apex_ms: tuple[float, float], seed: int, cal: dict
) -> np.ndarray:
    """Ringing ratios of simulated isolated spikes with apex width uniform in ``apex_ms``."""
    p = params.replace(
        duration_s=cal["spike_sim_duration_s"],
        events={
            "rate_ripple_per_min": 0.0,
            "rate_fast_ripple_per_min": 0.0,
            "rate_spike_per_min": cal["spike_sim_rate_per_min"],
        },
        spike={"apex_sigma_ms": apex_ms},
    )
    sim = simulate.simulate_from_seed(p, seed)
    high = filters.bandpass(sim.signal_uv, sim.fs, p.events.fast_ripple_band_hz, p.filter_order)
    scale = robust_std(high)
    half = int(cal["ringing_half_window_s"] * sim.fs)
    centres = (sim.events["center_s"].to_numpy() * sim.fs).round().astype(int)
    return np.array([ringing_ratio(high, c, half, scale) for c in centres])


def log_quantile_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared difference of log10 quantiles (10, 25, 50, 75, 90 %)."""
    qa, qb = np.quantile(a, RINGING_QUANTILES), np.quantile(b, RINGING_QUANTILES)
    return float(np.sqrt(np.mean((np.log10(qa) - np.log10(qb)) ** 2)))


def search_apex_range(
    params: SimulationParams, real: np.ndarray, seed: int, cal: dict
) -> pd.DataFrame:
    """Distance to the real ringing quantiles for every candidate (low, high) apex range.

    The simulated quantiles of every candidate are saved too (``sim_q10`` ... ``sim_q90``): they
    do not depend on any patient, so a leave-one-patient-out recalibration only needs to
    recompute the real quantiles without the test patient (scripts/10_evaluate_real.py).
    """
    rows = []
    for low in cal["apex_low_grid_ms"]:
        for high in cal["apex_high_grid_ms"]:
            if high < low:
                continue
            sim = simulated_ringing(params, (low, high), seed, cal)
            quantiles = np.quantile(sim, RINGING_QUANTILES)
            rows.append(
                {
                    "apex_low_ms": low,
                    "apex_high_ms": high,
                    "distance_log10": log_quantile_distance(sim, real),
                    **{
                        f"sim_q{int(q * 100)}": v
                        for q, v in zip(RINGING_QUANTILES, quantiles, strict=True)
                    },
                }
            )
    return pd.DataFrame(rows).sort_values("distance_log10", ignore_index=True)


def plot_spike_calibration(real: np.ndarray, sims: dict[str, np.ndarray], path: Path) -> None:
    """Empirical CDFs of the ringing ratio: real transients vs simulated spikes."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    styles = [(plotting.ACCENT, "--"), (plotting.SERIES[0], "-")]
    ax.step(
        np.sort(real),
        np.linspace(0, 1, len(real)),
        color=plotting.INK,
        lw=1.4,
        label=f"real transients (n={len(real)})",
    )
    for (label, values), (color, ls) in zip(sims.items(), styles, strict=True):
        ax.step(
            np.sort(values), np.linspace(0, 1, len(values)), color=color, ls=ls, lw=1.2, label=label
        )
    ax.set_xscale("log")
    ax.set_xlabel("Ringing ratio: max |250-500 Hz| near the transient / robust std")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("How strongly spikes ring after band-pass filtering", loc="left")
    ax.legend(loc="lower right", fontsize=8)
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
    seed = cfg["simulation"]["seed"]

    # Part 1: background spectrum.
    fits, freqs, real_psd = fit_real_channels(cfg)
    fits.round(5).to_csv(results / "background_fit.csv", index=False, lineterminator="\n")
    summary = summarise(fits, fit_range[1])
    summary.round(4).to_csv(
        results / "background_fit_summary.csv", index=False, lineterminator="\n"
    )
    sim = simulated_spectra(params, seed, SIMULATED_CHANNELS)
    plot(freqs, real_psd, fits, sim, params, fit_range, figures / "sim_background_calibration.png")
    print(summary.to_string(index=False))

    # Part 2: spike apex width.
    cal = cfg["calibration"]
    real = measure_real_ringing(cfg)
    real.round(4).to_csv(results / "spike_ringing_real.csv", index=False, lineterminator="\n")
    grid = search_apex_range(params, real["ratio"].to_numpy(), seed, cal)
    grid.round(4).to_csv(results / "spike_apex_grid.csv", index=False, lineterminator="\n")
    best = (grid.loc[0, "apex_low_ms"], grid.loc[0, "apex_high_ms"])
    configured = params.spike.apex_sigma_ms
    sims = {
        "simulated, pure cusp (apex 0 ms)": simulated_ringing(params, (0.0, 0.0), seed, cal),
        f"simulated, configured apex {configured[0]}-{configured[1]} ms": simulated_ringing(
            params, configured, seed, cal
        ),
    }
    quantiles = {"real": real["ratio"].to_numpy(), **sims}
    table = pd.DataFrame(
        {name: np.quantile(v, RINGING_QUANTILES) for name, v in quantiles.items()},
        index=[f"q{int(q * 100)}" for q in RINGING_QUANTILES],
    )
    table.round(3).to_csv(results / "spike_calibration_summary.csv", lineterminator="\n")
    plot_spike_calibration(real["ratio"].to_numpy(), sims, figures / "sim_spike_calibration.png")
    print(f"\n{len(real)} real transients in {real.patient.nunique()} patients")
    print(
        f"Best apex range: {best[0]}-{best[1]} ms (configured: {configured[0]}-{configured[1]} ms)"
    )
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
