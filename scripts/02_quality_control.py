"""Quality control of every BIDS recording: integrity, spectra, line noise, bad channels.

Usage:
    python scripts/02_quality_control.py [--config configs/default.toml]

Outputs:
    results/qc_duplicates.csv     channels whose samples are identical across recordings
    results/qc_channels.csv       one row per channel (QC metrics, flags, FR count)
    results/qc_recordings.csv     one row per recording (summary)
    figures/qc_psd_pre.png, figures/qc_psd_post.png   PSD of every channel, per patient
    figures/qc_line_noise.png     mains harmonics per recording (heatmap)
    figures/qc_harmonics_examples.png   two recordings with only even / only odd harmonics
    BIDS channels.tsv             status / status_description updated (bad channels marked)
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import mne_bids
import numpy as np
import pandas as pd

from epidetect import io, plotting, qc
from epidetect.config import load_config

DUPLICATE_REASON = "byte-identical to another patient's recording in the source data"


def load_all(bids_root: Path, task: str) -> dict[tuple[str, str], dict]:
    """Read every recording from BIDS: data in uV, channel names, FR count per channel."""
    patients = sorted(mne_bids.get_entity_vals(bids_root, "subject"))
    recordings = {}
    for patient in patients:
        for session in io.BIDS_SESSIONS:
            raw = io.read_bids_recording(bids_root, patient, session, task)
            recordings[(patient, session)] = {
                "data_uv": raw.get_data() / io.MICROVOLT,
                "fs": raw.info["sfreq"],
                "ch_names": list(raw.ch_names),
                "n_fr": io.annotations_to_markings(raw)["channel"].value_counts(),
            }
    return recordings


def find_duplicates(recordings: dict[tuple[str, str], dict]) -> pd.DataFrame:
    """Channels whose full sample array is identical in more than one recording (SHA-1)."""
    owners: dict[str, list[tuple[str, str, str]]] = {}
    for (patient, session), rec in recordings.items():
        for ch, row in zip(rec["ch_names"], rec["data_uv"], strict=True):
            digest = hashlib.sha1(row.astype(np.float32).tobytes()).hexdigest()
            owners.setdefault(digest, []).append((patient, session, ch))
    rows = [
        {"group": k, "patient": p, "session": s, "channel": c}
        for k, group in enumerate(g for g in owners.values() if len(g) > 1)
        for p, s, c in group
    ]
    return pd.DataFrame(rows, columns=["group", "patient", "session", "channel"])


def check_exclusions(excluded: set[tuple[str, str]], duplicates: pd.DataFrame) -> None:
    """Every excluded recording must be justified by the duplicate check."""
    dup_recordings = set(zip(duplicates["patient"], duplicates["session"], strict=True))
    unjustified = excluded - dup_recordings
    if unjustified:
        raise ValueError(f"Excluded recordings not found among duplicates: {unjustified}")
    unexpected = dup_recordings - excluded
    if unexpected:
        print(f"WARNING: duplicated recordings not excluded in the config: {sorted(unexpected)}")


def run_qc(
    recordings: dict[tuple[str, str], dict], params: qc.QCParams, excluded: set
) -> tuple[pd.DataFrame, dict]:
    """Per-channel QC table for all recordings, plus the PSDs for plotting."""
    tables, spectra = [], {}
    for (patient, session), rec in recordings.items():
        table, freqs, psd = qc.channel_qc(rec["data_uv"], rec["fs"], rec["ch_names"], params)
        table.insert(0, "session", session)
        table.insert(0, "patient", patient)
        table["n_fr"] = table["channel"].map(rec["n_fr"]).fillna(0).astype(int)
        table["excluded_recording"] = (patient, session) in excluded
        tables.append(table)
        spectra[(patient, session)] = (freqs, psd, table["bad"].to_numpy())
    return pd.concat(tables, ignore_index=True), spectra


def mark_bids_channels(bids_root: Path, table: pd.DataFrame, task: str) -> None:
    """Write QC flags to channels.tsv (reset to good first so the script is idempotent)."""
    for (patient, session), group in table.groupby(["patient", "session"]):
        bids_path = io.make_bids_path(bids_root, patient, session, task)
        names = group["channel"].tolist()
        mne_bids.mark_channels(bids_path, ch_names=names, status="good", verbose=False)
        if group["excluded_recording"].iloc[0]:
            bad, why = names, [DUPLICATE_REASON] * len(names)
        else:
            bad = group.loc[group["bad"], "channel"].tolist()
            why = group.loc[group["bad"], "reasons"].tolist()
        if bad:
            mne_bids.mark_channels(
                bids_path, ch_names=bad, status="bad", descriptions=why, verbose=False
            )


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """One row per recording."""
    line_cols = [c for c in table.columns if c.startswith("line_")]
    grouped = table.groupby(["patient", "session"])
    summary = grouped.agg(
        n_channels=("channel", "size"),
        n_flagged=("bad", "sum"),
        median_std_uv=("std_uv", "median"),
        median_aperiodic_beta=("aperiodic_beta", "median"),
        n_fr=("n_fr", "sum"),
        excluded=("excluded_recording", "first"),
    )
    summary["worst_harmonic_median_db"] = grouped[line_cols].median().max(axis=1)
    return summary.round(3).reset_index()


def plot_psd_grid(spectra: dict, session: str, excluded: set, cfg: dict, path: Path) -> None:
    """Small multiples: log-log PSD of every channel, one panel per patient."""
    keys = sorted(k for k in spectra if k[1] == session)
    n_cols = 6
    n_rows = int(np.ceil(len(keys) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(13, 2.1 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    fr_lo, fr_hi = cfg["qc"]["fast_ripple_band_hz"]
    r_lo, r_hi = cfg["qc"]["ripple_band_hz"]
    for ax, key in zip(axes.flat, keys, strict=False):
        freqs, psd, bad = spectra[key]
        keep = freqs >= 1.0
        ax.axvspan(r_lo, r_hi, color=plotting.BAND_RIPPLE, alpha=0.07, lw=0)
        ax.axvspan(fr_lo, fr_hi, color=plotting.BAND_FAST_RIPPLE, alpha=0.10, lw=0)
        for row, is_bad in zip(psd, bad, strict=True):
            if not is_bad:
                ax.loglog(freqs[keep], row[keep], color=plotting.NEUTRAL, lw=0.5, alpha=0.7)
        for row in psd[bad]:
            ax.loglog(freqs[keep], row[keep], color=plotting.ACCENT, lw=0.9)
        title = f"P{key[0]} ({psd.shape[0]} ch, {int(bad.sum())} flagged)"
        if key in excluded:
            title = f"P{key[0]}: duplicate, excluded"
            for line in ax.get_lines():
                line.set_alpha(0.25)
        ax.set_title(title, loc="left")
    for ax in axes.flat[len(keys) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("PSD (uV²/Hz)")
    handles = [
        plt.Line2D([], [], color=plotting.NEUTRAL, lw=1, label="channel"),
        plt.Line2D([], [], color=plotting.ACCENT, lw=1.5, label="flagged by QC"),
        plt.Rectangle((0, 0), 1, 1, color=plotting.BAND_RIPPLE, alpha=0.15, label="ripple band"),
        plt.Rectangle(
            (0, 0), 1, 1, color=plotting.BAND_FAST_RIPPLE, alpha=0.2, label="fast-ripple band"
        ),
    ]
    fig.legend(handles=handles, loc="upper right", ncols=4, bbox_to_anchor=(1.0, 1.02))
    label = "Pre-resection" if session == "pre" else "Post-resection"
    fig.suptitle(f"{label}: median-Welch PSD of every bipolar channel", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    plotting.save(fig, path)


def plot_line_noise(table: pd.DataFrame, present_db: float, path: Path) -> None:
    """Heatmap: median (over channels) peak height of each 50 Hz harmonic, per recording."""
    line_cols = [c for c in table.columns if c.startswith("line_")]
    harmonics = [int(c.split("_")[1]) for c in line_cols]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
    vmax = 20.0  # dB; a few cells exceed it and saturate
    for ax, session in zip(axes, ["pre", "post"], strict=True):
        med = table[table.session == session].groupby("patient")[line_cols].median()
        image = ax.imshow(
            med.to_numpy().clip(min=0),
            cmap=plotting.SEQUENTIAL_CMAP,
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        for (i, j), value in np.ndenumerate(med.to_numpy()):
            if value > present_db:  # label only harmonics clearly present
                color = "white" if value > vmax * 0.6 else plotting.INK
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7, color=color)
        ax.set_xticks(range(len(harmonics)), harmonics)
        ax.set_yticks(range(len(med)), [f"P{p}" for p in med.index])
        ax.set_xlabel("Harmonic of 50 Hz (Hz)")
        ax.set_title("Pre-resection" if session == "pre" else "Post-resection", loc="left")
        ax.grid(False)
    cbar = fig.colorbar(image, ax=axes, shrink=0.8)
    cbar.set_label("Peak above its neighbourhood (dB, median over channels)")
    fig.suptitle(
        f"Mains interference per recording (values shown when > {present_db:.0f} dB)",
        x=0.01,
        ha="left",
    )
    plotting.save(fig, path)


def plot_harmonic_examples(recordings: dict, table: pd.DataFrame, cfg: dict, path: Path) -> None:
    """Two recordings whose mains interference has only even / only odd harmonics."""
    cases = [
        (("14", "pre"), "P14 pre-resection: only even harmonics (100, 200, 300, 400 Hz)"),
        (("20", "post"), "P20 post-resection: only odd harmonics (150, 250, 350, 450 Hz)"),
    ]
    line_cols = [c for c in table.columns if c.startswith("line_")]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    for ax, (key, title) in zip(axes, cases, strict=True):
        rec = recordings[key]
        rows = table[(table.patient == key[0]) & (table.session == key[1])]
        best = int(rows[line_cols].max(axis=1).to_numpy().argmax())  # most affected channel
        qc_cfg = cfg["qc"]
        freqs, psd = qc.welch_psd(
            rec["data_uv"][best : best + 1],
            rec["fs"],
            qc_cfg["welch_window_s"],
            qc_cfg["welch_overlap"],
            average="mean",
        )
        keep = (freqs >= 20) & (freqs <= 500)
        for h in qc.line_harmonics(50.0, 450.0):
            is_even = int(h / 50) % 2 == 0
            ax.axvline(
                h,
                color=plotting.SERIES[0] if is_even else plotting.SERIES[1],
                lw=0.8,
                ls="--",
                alpha=0.6,
            )
        ax.semilogy(freqs[keep], psd[0, keep], color=plotting.INK, lw=0.7)
        ax.set_title(f"{title}, channel {rec['ch_names'][best]}", loc="left")
        ax.set_ylabel("PSD (uV²/Hz)")
    axes[-1].set_xlabel("Frequency (Hz)")
    handles = [
        plt.Line2D([], [], color=plotting.SERIES[0], ls="--", label="even multiple of 50 Hz"),
        plt.Line2D([], [], color=plotting.SERIES[1], ls="--", label="odd multiple of 50 Hz"),
    ]
    fig.legend(handles=handles, loc="upper right", ncols=2, bbox_to_anchor=(1.0, 1.03))
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    bids_root = cfg["paths"]["bids_root"]
    results, figures = cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"]
    results.mkdir(exist_ok=True)
    task = cfg["bids"]["task"]
    excluded = {tuple(x) for x in cfg["data"]["excluded_recordings"]}
    params = qc.QCParams.from_config(cfg["qc"], cfg["recording"]["line_freq_hz"])

    recordings = load_all(bids_root, task)
    duplicates = find_duplicates(recordings)
    duplicates.to_csv(results / "qc_duplicates.csv", index=False, lineterminator="\n")
    check_exclusions(excluded, duplicates)

    table, spectra = run_qc(recordings, params, excluded)
    table.round(4).to_csv(results / "qc_channels.csv", index=False, lineterminator="\n")
    summary = summarize(table)
    summary.to_csv(results / "qc_recordings.csv", index=False, lineterminator="\n")
    mark_bids_channels(bids_root, table, task)

    plot_psd_grid(spectra, "pre", excluded, cfg, figures / "qc_psd_pre.png")
    plot_psd_grid(spectra, "post", excluded, cfg, figures / "qc_psd_post.png")
    plot_line_noise(table, params.line_present_db, figures / "qc_line_noise.png")
    plot_harmonic_examples(recordings, table, cfg, figures / "qc_harmonics_examples.png")

    analysed = table[~table.excluded_recording]
    print(f"Duplicated channels: {len(duplicates)} in {duplicates.group.nunique()} groups")
    print(f"Channels analysed: {len(analysed)}; flagged: {int(analysed.bad.sum())}")
    print(f"Flagged channels carrying reference FRs: {int(analysed[analysed.bad].n_fr.sum())}")
    print(
        "Median aperiodic exponent beta: "
        f"{analysed.aperiodic_beta.median():.2f} "
        f"(IQR {analysed.aperiodic_beta.quantile(0.25):.2f}-"
        f"{analysed.aperiodic_beta.quantile(0.75):.2f})"
    )
    print(
        analysed.loc[analysed.bad, ["patient", "session", "channel", "n_fr", "reasons"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
