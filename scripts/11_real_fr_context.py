"""How many real reference fast ripples ride on a large transient (spike)?

Same transient definition as the spike calibration (``[calibration]``): robust z-score above
``transient_z`` in the 3-40 Hz band. A reference FR is "on a transient" if such a peak lies
within ``WINDOW_S`` of its centre, on the same channel. This explains why the wavelet detector,
whose spike-rejection test also rejects HFOs on spikes in simulation, finds few real FRs.

Usage:
    python scripts/11_real_fr_context.py [--config configs/default.toml]

Output:
    results/real_fr_context.csv   per patient: number of FRs and fraction on a transient
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from epidetect import filters, io
from epidetect.config import load_config
from epidetect.qc import robust_std

WINDOW_S = 0.05  # transient peak within +/- 50 ms of the FR centre


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cal = cfg["calibration"]
    excluded = {tuple(x) for x in cfg["data"]["excluded_recordings"]}
    rows = []
    for patient in sorted(io.list_patient_files(cfg["paths"]["raw_dir"])):
        if (patient, "pre") in excluded:
            continue
        raw = io.read_bids_recording(cfg["paths"]["bids_root"], patient, "pre", cfg["bids"]["task"])
        fs = raw.info["sfreq"]
        markings = io.annotations_to_markings(raw)
        if markings.empty:
            continue
        n_on = 0
        for channel, group in markings.groupby("channel"):
            x = raw.get_data(picks=[channel])[0] / io.MICROVOLT
            low = filters.bandpass(x, fs, tuple(cal["transient_band_hz"]))
            z = np.abs((low - np.median(low)) / robust_std(low))
            peaks, _ = find_peaks(
                z, height=cal["transient_z"], distance=int(cal["transient_min_gap_s"] * fs)
            )
            peak_times = peaks / fs
            centres = (group.onset_s + group.duration_s / 2).to_numpy()
            if peak_times.size:
                distance = np.abs(centres[:, None] - peak_times[None, :]).min(axis=1)
                n_on += int((distance <= WINDOW_S).sum())
        rows.append(
            {
                "patient": patient,
                "n_fr": len(markings),
                "n_on_transient": n_on,
                "fraction_on_transient": n_on / len(markings),
            }
        )
    table = pd.DataFrame(rows)
    table.round(4).to_csv(
        cfg["paths"]["results_dir"] / "real_fr_context.csv", index=False, lineterminator="\n"
    )
    print(table.round(2).to_string(index=False))
    print(
        f"\nAll patients pooled: {table.n_on_transient.sum()} / {table.n_fr.sum()} = "
        f"{table.n_on_transient.sum() / table.n_fr.sum():.2f}; "
        f"median over patients: {table.fraction_on_transient.median():.2f}"
    )


if __name__ == "__main__":
    main()
