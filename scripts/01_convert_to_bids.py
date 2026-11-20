"""Convert the 22 raw NIX files (pre- and post-resection) to a BIDS iEEG dataset.

Usage:
    python scripts/01_convert_to_bids.py [--config configs/default.toml]

Output: ``data/bids/`` (git-ignored) and ``results/bids_conversion.csv`` (one row per recording,
used to check that nothing was lost: channels, duration and FR counts must match the inventory).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from epidetect import io
from epidetect.config import load_config

BIDS_README = """\
Intraoperative ECoG (pre- and post-resection) with fast-ripple markings, converted to BIDS.

Source: https://gin.g-node.org/USZ_NCH/Intraoperative_ECoG_HFO (licence CC BY-SA 4.0).
Converted with the epidetect project (scripts/01_convert_to_bids.py).

Conversion notes:
- Signals were already in a bipolar montage in the source; channel names are contact pairs.
- No electrode coordinates exist, so electrodes.tsv contains n/a coordinates.
- FR markings are in events.tsv with an extra `channel` column (described in events.json).
- The source does not give which channels lie over resected tissue; `resection_area` in
  participants.tsv is free text.
"""


def verify_round_trip(bids_root: Path, rec: io.Recording, task: str) -> float:
    """Re-read one recording from BIDS and check it against the source.

    Returns:
        Maximum absolute error in microvolts (float32 storage gives ~1e-7 relative error).

    Raises:
        AssertionError: If channels or FR markings differ.
    """
    raw = io.read_bids_recording(bids_root, rec.patient, rec.session, task)
    assert raw.ch_names == rec.ch_names, f"channel mismatch for {rec.patient} {rec.session}"
    back = io.annotations_to_markings(raw)
    assert len(back) == len(rec.markings), f"FR count mismatch for {rec.patient} {rec.session}"
    assert sorted(back["channel"]) == sorted(rec.markings["channel"])
    return float(abs(raw.get_data() / io.MICROVOLT - rec.data_uv).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_dir, bids_root = cfg["paths"]["raw_dir"], cfg["paths"]["bids_root"]
    rec_cfg, bids_cfg = cfg["recording"], cfg["bids"]

    if bids_root.exists():  # rebuild from scratch: the conversion must be fully reproducible
        shutil.rmtree(bids_root)
    bids_root.mkdir(parents=True)

    rows, metadata = [], {}
    for patient, path in io.list_patient_files(raw_dir).items():
        metadata[patient] = io.read_metadata(path)
        for session in io.NIX_BLOCKS:
            rec = io.read_nix_recording(path, patient, session)
            bids_path = io.write_recording_to_bids(
                rec,
                bids_root,
                task=bids_cfg["task"],
                line_freq_hz=rec_cfg["line_freq_hz"],
                highpass_hz=rec_cfg["hardware_highpass_hz"],
                lowpass_hz=rec_cfg["hardware_lowpass_hz"],
            )
            max_err_uv = verify_round_trip(bids_root, rec, bids_cfg["task"])
            rows.append(
                {
                    "patient": patient,
                    "session": session,
                    "bids_file": bids_path.basename,
                    "n_channels": len(rec.ch_names),
                    "duration_s": round(rec.duration_s, 3),
                    "n_fr": len(rec.markings),
                    "max_abs_error_uv": max_err_uv,
                }
            )
            print(
                f"sub-{patient} {session:4s}: {len(rec.ch_names):2d} ch, "
                f"{len(rec.markings):4d} FR, round-trip max error {max_err_uv:.2e} uV"
            )

    io.write_participants(bids_root, metadata)
    io.write_dataset_description(bids_root)
    (bids_root / "README").write_text(BIDS_README, encoding="utf-8", newline="\n")

    results_dir = cfg["paths"]["results_dir"]
    results_dir.mkdir(exist_ok=True)
    out = results_dir / "bids_conversion.csv"
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(f"Wrote {bids_root} and {out}")


if __name__ == "__main__":
    main()
