"""Figure: the three detectors on four typical simulated situations.

Columns: isolated ripple, isolated fast ripple, isolated spike (no HFO), fast ripple on a spike.
Rows: raw signal; band-passed signal with the ground truth and each detector's detections;
whitened Morlet map (robust z-score), where an HFO is an isolated blob and a spike a column.

The ML detector is trained on simulations with seeds ``TRAIN_SEEDS``, all different from the
example simulation. This figure illustrates behaviours; the quantitative benchmark is step 4.

Usage:
    python scripts/07_detector_examples.py [--config configs/default.toml]

Output:
    figures/detector_examples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from epidetect import filters, plotting, simulate
from epidetect.config import load_config
from epidetect.detectors import bands_from_config
from epidetect.detectors.energy import EnergyParams, detect_energy
from epidetect.detectors.ml import MLDetector, MLParams
from epidetect.detectors.wavelet import WaveletParams, detect_wavelet, iter_whitened_chunks
from epidetect.simulate import SimulationParams

EXAMPLE_DURATION_S = 300.0
TRAIN_SEEDS = (101, 102, 103, 104)  # different from [simulation] seed (the example)
TRAIN_SNR_DB, TRAIN_SNR_JITTER_DB = 8.0, 5.0  # training covers a range of SNRs
HALF_WINDOW_S = 0.12
DETECTOR_COLORS = {
    "energy": plotting.SERIES[0],
    "wavelet": plotting.SERIES[1],
    "ML": plotting.SERIES[2],
}


def pick(events, kind: str, on_spike: bool):
    rows = events[(events.kind == kind) & (events.on_spike == on_spike)]
    if rows.empty:
        raise RuntimeError(f"No {kind} (on_spike={on_spike}) in the example simulation.")
    return rows.iloc[0]


def whitened_window(x: np.ndarray, fs: float, wp: WaveletParams, centre_s: float, edge_s: float):
    """Whitened map around ``centre_s`` (taken from the chunk that contains it)."""
    centre = int(centre_s * fs)
    for chunk in iter_whitened_chunks(x, fs, wp, edge_s):
        if chunk.core_start <= centre < chunk.core_stop:
            half = int(HALF_WINDOW_S * fs)
            start = centre - half - chunk.seg_start
            return chunk.z[:, start : start + 2 * half]
    raise ValueError("centre outside the analysed part of the signal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    det_cfg = cfg["detectors"]
    bands = bands_from_config(det_cfg)
    edge = det_cfg["edge_margin_s"]
    energy_p = EnergyParams.from_config(det_cfg["energy"])
    wavelet_p = WaveletParams.from_config(det_cfg["wavelet"])
    ml_p = MLParams.from_config(det_cfg["ml"], cfg["features"])
    base = SimulationParams.from_config(cfg["simulation"]).replace(duration_s=EXAMPLE_DURATION_S)

    train = [
        simulate.simulate_from_seed(
            base.replace(events={"snr_db": TRAIN_SNR_DB, "snr_jitter_db": TRAIN_SNR_JITTER_DB}), s
        )
        for s in TRAIN_SEEDS
    ]
    ml = MLDetector(bands, ml_p, energy_p, wavelet_p, edge).fit(
        [(s.signal_uv, s.fs, s.events) for s in train]
    )
    sim = simulate.simulate_from_seed(base, cfg["simulation"]["seed"])
    x, fs = sim.signal_uv, sim.fs
    detections = {
        "energy": detect_energy(x, fs, bands, energy_p, edge),
        "wavelet": detect_wavelet(x, fs, bands, wavelet_p, edge),
        "ML": ml.detect(x, fs),
    }
    examples = {
        "Ripple": pick(sim.events, "ripple", False),
        "Fast ripple": pick(sim.events, "fast_ripple", False),
        "Isolated spike (no HFO)": pick(sim.events, "spike", False),
        "Fast ripple on a spike": pick(sim.events, "fast_ripple", True),
    }
    filtered = {name: filters.bandpass(x, fs, band) for name, band in bands.items()}
    freqs = wavelet_p.freqs()

    fig, axes = plt.subplots(
        3, 4, figsize=(15, 8.5), sharex="col", gridspec_kw={"height_ratios": [1, 1.2, 1.3]}
    )
    fig.subplots_adjust(wspace=0.35, hspace=0.3)
    for col, (title, ev) in enumerate(examples.items()):
        half = int(HALF_WINDOW_S * fs)
        centre = int(ev.center_s * fs)
        sl = slice(centre - half, centre + half)
        t_ms = (np.arange(sl.start, sl.stop) / fs - ev.center_s) * 1000
        band = "ripple" if ev.kind == "ripple" else "fast_ripple"
        truth = ((ev.onset_s - ev.center_s) * 1000, (ev.offset_s - ev.center_s) * 1000)
        detail = (
            f"{ev.freq_hz:.0f} Hz, SNR {ev.snr_db:.0f} dB"
            if ev.kind != "spike"
            else f"{ev.spike_amplitude_uv:.0f} uV"
        )

        ax = axes[0, col]
        ax.plot(t_ms, x[sl], color=plotting.INK, lw=0.6)
        ax.set_title(f"{title}\n{detail}", loc="left")
        ax.set_ylabel("raw (uV)")

        ax = axes[1, col]
        y = filtered[band][sl]
        ax.axvspan(*truth, color=plotting.NEUTRAL, alpha=0.25, lw=0)
        ax.plot(t_ms, y, color=plotting.INK, lw=0.6)
        top = np.abs(y).max()
        for k, (name, det) in enumerate(detections.items()):
            hits = det[(det.offset_s > sl.start / fs) & (det.onset_s < sl.stop / fs)]
            for row in hits.itertuples():
                ax.plot(
                    [(row.onset_s - ev.center_s) * 1000, (row.offset_s - ev.center_s) * 1000],
                    [top * (1.15 + 0.15 * k)] * 2,
                    color=DETECTOR_COLORS[name],
                    lw=3,
                    solid_capstyle="butt",
                )
        ax.set_ylim(-1.1 * top, top * 1.65)
        low, high = bands[band]
        ax.set_ylabel(f"{low:.0f}-{high:.0f} Hz (uV)")

        ax = axes[2, col]
        z = whitened_window(x, fs, wavelet_p, ev.center_s, edge)
        mesh = ax.pcolormesh(
            t_ms, freqs, z, cmap=plotting.SEQUENTIAL_CMAP, vmin=0, vmax=20, shading="auto"
        )
        ax.set_yscale("log")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time from event centre (ms)")
        ax.grid(False)
    # Colour bar in an inset to the right of the last panel: the panels keep equal widths.
    cax = axes[2, -1].inset_axes([1.04, 0.0, 0.04, 1.0])
    fig.colorbar(mesh, cax=cax, label="whitened z-score")
    handles = [plt.Rectangle((0, 0), 1, 1, color=plotting.NEUTRAL, alpha=0.4, label="ground truth")]
    handles += [
        plt.Line2D([], [], color=c, lw=3, label=f"{n} detection")
        for n, c in DETECTOR_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper right", ncols=4, bbox_to_anchor=(0.9, 0.99))
    plotting.save(fig, cfg["paths"]["figures_dir"] / "detector_examples.png")
    for name, det in detections.items():
        print(f"{name}: {len(det)} detections in {EXAMPLE_DURATION_S:.0f} s")


if __name__ == "__main__":
    main()
