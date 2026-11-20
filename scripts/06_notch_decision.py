"""Should mains harmonics be notch-filtered before HFO detection? Measure it on the real data.

For every analysed channel whose QC shows at least one mains harmonic inside the HFO bands
(100-450 Hz, peak above ``[qc] line_present_db``), the energy and wavelet detectors are run on
the raw signal and on the signal with those harmonics notched out. We count detections and
"harmonic-locked" detections (estimated frequency within ``LOCK_TOLERANCE_HZ`` of a present
harmonic).

The decision deliberately ignores the reference FR markings: they are kept for the evaluation
of step 4, so using them here would leak evaluation data into a pre-processing choice.

Usage:
    python scripts/06_notch_decision.py [--config configs/default.toml]

Output:
    results/notch_decision.csv   one row per (channel, detector)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epidetect import filters, io
from epidetect.config import load_config
from epidetect.detectors import bands_from_config
from epidetect.detectors.energy import EnergyParams, detect_energy
from epidetect.detectors.wavelet import WaveletParams, detect_wavelet

LOCK_TOLERANCE_HZ = 5.0  # a detection this close to a present harmonic is "harmonic-locked"
HFO_HARMONIC_MIN_HZ = 80.0  # harmonics below the ripple band are irrelevant here
NOTCH_QUALITY = 30.0  # Q = f / bandwidth (see epidetect.filters.notch_harmonics)


def present_harmonics(qc_row: pd.Series, present_db: float) -> list[float]:
    """Harmonics in the HFO bands whose QC peak exceeds ``present_db``."""
    out = []
    for column, value in qc_row.items():
        if str(column).startswith("line_") and str(column).endswith("_db"):
            freq = float(str(column).split("_")[1])
            if freq >= HFO_HARMONIC_MIN_HZ and value > present_db:
                out.append(freq)
    return out


def locked(detections: pd.DataFrame, harmonics: list[float]) -> int:
    """Number of detections whose frequency is within the tolerance of a harmonic."""
    if detections.empty:
        return 0
    freqs = detections["freq_hz"].to_numpy(float)[:, None]
    return int(
        (np.abs(freqs - np.array(harmonics)[None, :]) <= LOCK_TOLERANCE_HZ).any(axis=1).sum()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    det_cfg = cfg["detectors"]
    bands = bands_from_config(det_cfg)
    edge = det_cfg["edge_margin_s"]
    energy_params = EnergyParams.from_config(det_cfg["energy"])
    wavelet_params = WaveletParams.from_config(det_cfg["wavelet"])
    qc_table = pd.read_csv(cfg["paths"]["results_dir"] / "qc_channels.csv", dtype={"patient": str})
    qc_table = qc_table[~qc_table["excluded_recording"] & ~qc_table["bad"]]

    rows = []
    for (patient, session), group in qc_table.groupby(["patient", "session"]):
        affected = {
            r.channel: present_harmonics(r, cfg["qc"]["line_present_db"])
            for _, r in group.iterrows()
        }
        affected = {ch: h for ch, h in affected.items() if h}
        if not affected:
            continue
        raw = io.read_bids_recording(
            cfg["paths"]["bids_root"], patient, session, cfg["bids"]["task"]
        )
        fs = raw.info["sfreq"]
        for channel, harmonics in affected.items():
            x = raw.get_data(picks=[channel])[0] / io.MICROVOLT
            x_notched = filters.notch_harmonics(x, fs, harmonics, NOTCH_QUALITY)
            for name, detect, params in [
                ("energy", detect_energy, energy_params),
                ("wavelet", detect_wavelet, wavelet_params),
            ]:
                before = detect(x, fs, bands, params, edge)
                after = detect(x_notched, fs, bands, params, edge)
                rows.append(
                    {
                        "patient": patient,
                        "session": session,
                        "channel": channel,
                        "harmonics_hz": " ".join(f"{h:.0f}" for h in harmonics),
                        "detector": name,
                        "n_raw": len(before),
                        "n_notched": len(after),
                        "locked_raw": locked(before, harmonics),
                        "locked_notched": locked(after, harmonics),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(
        cfg["paths"]["results_dir"] / "notch_decision.csv", index=False, lineterminator="\n"
    )
    summary = table.groupby("detector")[
        ["n_raw", "n_notched", "locked_raw", "locked_notched"]
    ].sum()
    n_channels = table[["patient", "session", "channel"]].drop_duplicates().shape[0]
    n_recordings = table[["patient", "session"]].drop_duplicates().shape[0]
    print(f"{n_channels} channels with harmonics in the HFO bands, in {n_recordings} recordings")
    print(summary.to_string())


if __name__ == "__main__":
    main()
