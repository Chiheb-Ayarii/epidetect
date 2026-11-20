"""Figures of simulated signals: one example per event type, and one event at several SNRs.

Usage:
    python scripts/04_simulation_examples.py [--config configs/default.toml]

Outputs:
    figures/sim_examples.png       ripple, fast ripple, isolated spike, fast ripple on a spike:
                                   raw signal and band-passed signal, ground truth shaded
    figures/sim_snr_levels.png     the same simulation at several SNRs (band-passed, 250-500 Hz)
    results/sim_example_truth.csv  ground truth of the example simulation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epidetect import filters, plotting, simulate
from epidetect.config import load_config
from epidetect.simulate import Simulation, SimulationParams

HALF_WINDOW_S = 0.15  # each panel shows +/- 150 ms around the event centre
EXAMPLE_DURATION_S = 300.0  # long enough to contain every event type (incl. HFO on a spike)
SNR_LEVELS_DB = (-5.0, 0.0, 5.0, 10.0, 15.0)


def pick_examples(events: pd.DataFrame) -> dict[str, pd.Series]:
    """First event of each type: ripple, fast ripple, isolated spike, fast ripple on spike."""
    hfo = events[events.kind != "spike"]
    choices = {
        "Ripple": hfo[(hfo.kind == "ripple") & ~hfo.on_spike],
        "Fast ripple": hfo[(hfo.kind == "fast_ripple") & ~hfo.on_spike],
        "Isolated spike (no HFO)": events[events.kind == "spike"],
        "Fast ripple on a spike": hfo[(hfo.kind == "fast_ripple") & hfo.on_spike],
    }
    missing = [name for name, rows in choices.items() if rows.empty]
    if missing:
        raise RuntimeError(f"No example of {missing}: lengthen the simulation or change the seed.")
    return {name: rows.iloc[0] for name, rows in choices.items()}


def window(sim: Simulation, centre_s: float) -> tuple[slice, np.ndarray]:
    """Sample slice and time axis (ms, relative to the centre) of a panel."""
    start = int((centre_s - HALF_WINDOW_S) * sim.fs)
    stop = int((centre_s + HALF_WINDOW_S) * sim.fs)
    return slice(start, stop), (np.arange(start, stop) / sim.fs - centre_s) * 1000


def plot_examples(sim: Simulation, path: Path) -> None:
    p = sim.params
    examples = pick_examples(sim.events)
    filtered = {
        "ripple": filters.bandpass(sim.signal_uv, sim.fs, p.events.ripple_band_hz, p.filter_order),
        "fast_ripple": filters.bandpass(
            sim.signal_uv, sim.fs, p.events.fast_ripple_band_hz, p.filter_order
        ),
    }
    fig, axes = plt.subplots(2, 4, figsize=(14, 5), sharex=True)
    for col, (name, ev) in enumerate(examples.items()):
        sl, t_ms = window(sim, ev.center_s)
        band_kind = "ripple" if ev.kind == "ripple" else "fast_ripple"
        band = p.events.ripple_band_hz if band_kind == "ripple" else p.events.fast_ripple_band_hz
        for row, (y, label) in enumerate(
            [
                (sim.signal_uv[sl], "raw"),
                (filtered[band_kind][sl], f"{band[0]:.0f}-{band[1]:.0f} Hz"),
            ]
        ):
            ax = axes[row, col]
            ax.axvspan(
                (ev.onset_s - ev.center_s) * 1000,
                (ev.offset_s - ev.center_s) * 1000,
                color=plotting.ACCENT,
                alpha=0.15,
                lw=0,
            )
            ax.plot(t_ms, y, color=plotting.INK, lw=0.6)
            ax.set_ylabel(f"{label} (uV)")
            if row == 0:
                detail = (
                    f"{ev.freq_hz:.0f} Hz, {ev.n_cycles:.1f} cycles, SNR {ev.snr_db:.0f} dB"
                    if ev.kind != "spike"
                    else f"amplitude {ev.spike_amplitude_uv:.0f} uV"
                )
                ax.set_title(f"{name}\n{detail}", loc="left")
            else:
                ax.set_xlabel("Time from event centre (ms)")
    handle = plt.Rectangle((0, 0), 1, 1, color=plotting.ACCENT, alpha=0.3, label="ground truth")
    fig.legend(handles=[handle], loc="upper right", bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout()
    plotting.save(fig, path)


def plot_snr_levels(params: SimulationParams, seed: int, path: Path) -> None:
    """Same seed, different SNR: identical events and background, only amplitudes change."""
    sims = [
        simulate.simulate_from_seed(params.replace(events={"snr_db": snr}), seed)
        for snr in SNR_LEVELS_DB
    ]
    events = sims[0].events
    target = events[(events.kind == "fast_ripple") & ~events.on_spike].iloc[0]
    band = params.events.fast_ripple_band_hz
    fig, axes = plt.subplots(1, len(SNR_LEVELS_DB), figsize=(14, 2.6), sharex=True, sharey=True)
    for ax, sim, snr in zip(axes, sims, SNR_LEVELS_DB, strict=True):
        y = filters.bandpass(sim.signal_uv, sim.fs, band, params.filter_order)
        sl, t_ms = window(sim, target.center_s)
        ax.axvspan(
            (target.onset_s - target.center_s) * 1000,
            (target.offset_s - target.center_s) * 1000,
            color=plotting.ACCENT,
            alpha=0.15,
            lw=0,
        )
        ax.plot(t_ms, y[sl], color=plotting.INK, lw=0.6)
        ax.set_title(f"SNR {snr:+.0f} dB", loc="left")
        ax.set_xlabel("Time (ms)")
    axes[0].set_ylabel(f"{band[0]:.0f}-{band[1]:.0f} Hz (uV)")
    fig.suptitle(
        f"One fast ripple ({target.freq_hz:.0f} Hz, {target.n_cycles:.1f} cycles) at increasing "
        "SNR, same background",
        x=0.01,
        ha="left",
        y=1.04,
    )
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    params = SimulationParams.from_config(cfg["simulation"]).replace(duration_s=EXAMPLE_DURATION_S)
    seed = cfg["simulation"]["seed"]
    sim = simulate.simulate_from_seed(params, seed)
    results, figures = cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"]
    sim.events.round(6).to_csv(results / "sim_example_truth.csv", index=False, lineterminator="\n")
    plot_examples(sim, figures / "sim_examples.png")
    plot_snr_levels(params, seed, figures / "sim_snr_levels.png")
    counts = sim.events.groupby(["kind", "on_spike"]).size()
    print(f"{params.duration_s:.0f} s simulated, {len(sim.events)} events:\n{counts.to_string()}")
    print("Background band power (uV^2):", {k: round(v, 4) for k, v in sim.band_power_uv2.items()})


if __name__ == "__main__":
    main()
