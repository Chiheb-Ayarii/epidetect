"""Real-data helpers shared by the evaluation scripts: channels, cached analyses, transients."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from epidetect import benchmark, filters, io
from epidetect.qc import robust_std

FR_BAND_NAME = "fast_ripple"


def channel_table(cfg: dict[str, Any], session: str = "pre") -> pd.DataFrame:
    """Channels of one session (``"pre"`` or ``"post"``) to analyse, with their QC flag.

    Excluded recordings are skipped. Channels with repeated samples (effective sampling rate
    below fs, e.g. P22 3-4) cannot carry fast ripples and are dropped.
    """
    qc = pd.read_csv(cfg["paths"]["results_dir"] / "qc_channels.csv", dtype={"patient": str})
    qc = qc[(qc.session == session) & ~qc.excluded_recording]
    unusable = qc.reasons.fillna("").str.contains("repeated samples")
    return qc.loc[~unusable, ["patient", "channel", "bad"]].reset_index(drop=True)


def marking_reference(markings: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Reference FR intervals of one channel in the format used by the detectors' scoring."""
    m = markings[markings.channel == channel]
    return pd.DataFrame(
        {"onset_s": m.onset_s, "offset_s": m.onset_s + m.duration_s, "kind": FR_BAND_NAME}
    )


def analyse_patient(
    cfg: dict[str, Any],
    patient: str,
    channels: list[str],
    settings: benchmark.DetectorSettings,
    session: str = "pre",
) -> tuple[benchmark.TrainingUnit, dict[str, float]]:
    """All detector outputs of one patient's channels in one session, and minutes per channel."""
    raw = io.read_bids_recording(cfg["paths"]["bids_root"], patient, session, cfg["bids"]["task"])
    fs = raw.info["sfreq"]
    markings = io.annotations_to_markings(raw)
    per_channel, minutes = {}, {}
    for channel in channels:
        x = raw.get_data(picks=[channel])[0] / io.MICROVOLT
        reference = marking_reference(markings, channel)
        per_channel[channel] = benchmark.analyse_signal(x, fs, reference, settings)[FR_BAND_NAME]
        minutes[channel] = (len(x) / fs - 2 * settings.edge_margin_s) / 60
    return benchmark.merge_channels(per_channel, patient), minutes


def load_real_units(
    cfg: dict[str, Any],
    settings: benchmark.DetectorSettings,
    channels: pd.DataFrame,
    session: str = "pre",
) -> tuple[dict[str, benchmark.TrainingUnit], dict[str, dict[str, float]]]:
    """Cached ``analyse_patient`` for every patient of ``channels``."""
    cache = benchmark.cache_dir(cfg, "real" if session == "pre" else f"real-{session}")
    units, minutes = {}, {}
    for patient, group in channels.groupby("patient"):
        units[patient], minutes[patient] = benchmark.load_or_compute(
            cache / f"{patient}.pkl",
            lambda p=patient, g=group: analyse_patient(
                cfg, p, g.channel.tolist(), settings, session
            ),
        )
    return units, minutes


def find_transients(
    x_uv: np.ndarray,
    fs: float,
    band: tuple[float, float],
    z_threshold: float,
    min_gap_s: float,
    order: int = 4,
) -> np.ndarray:
    """Times (s) of large transients: robust z-score of the band-passed signal above threshold.

    Same definition as the spike calibration of step 2 (3-40 Hz, |z| > 8, 0.3 s apart).
    """
    low = filters.bandpass(x_uv, fs, band, order)
    z = np.abs((low - np.median(low)) / robust_std(low))
    peaks, _ = find_peaks(z, height=z_threshold, distance=max(1, int(min_gap_s * fs)))
    return peaks / fs
