"""Load the Zurich intraoperative ECoG files (NIX/HDF5) and convert them to BIDS.

Raw file layout and its quirks are documented in ``docs/DATA_NOTES.md``. The two quirks handled
here are:

* fast-ripple (FR) tags store a **1-based** channel index (MATLAB style), not a 0-based NIX index;
* some metadata strings are stored one character per record.

Downstream code should read the BIDS dataset with :func:`read_bids_recording`, which restores
channel-specific FR annotations from the extra ``channel`` column of ``events.tsv``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import mne
import mne_bids
import numpy as np
import pandas as pd

# --- Raw NIX layout -------------------------------------------------------------------------
NIX_BLOCKS = {
    "pre": "Data_Pre_Resection_Bipolar_Montage",
    "post": "Data_Post_Resection_Bipolar_Montage",
}
SIGNAL_ARRAY = "ECoG_Bipolar_Channels"
FR_TAGS = "groups/FR markings/tags"
FR_DESCRIPTION = "FR"

# --- BIDS naming ----------------------------------------------------------------------------
BIDS_SESSIONS = {"pre": "preresection", "post": "postresection"}
MICROVOLT = 1e-6  # MNE stores EEG/ECoG data in volts; the raw files are in microvolts.


@dataclass(frozen=True)
class Recording:
    """One recording block (pre- or post-resection) of one patient.

    Attributes:
        patient: Two-digit patient identifier, e.g. ``"01"``.
        session: ``"pre"`` or ``"post"``.
        data_uv: Signals in microvolts, shape ``(n_channels, n_samples)``.
        fs: Sampling frequency in Hz.
        ch_names: Bipolar channel names, e.g. ``"01-02"``.
        markings: One row per FR with columns ``channel`` (name), ``onset_s``, ``duration_s``.
    """

    patient: str
    session: str
    data_uv: np.ndarray
    fs: float
    ch_names: list[str]
    markings: pd.DataFrame = field(repr=False)

    @property
    def duration_s(self) -> float:
        """Recording duration in seconds."""
        return self.data_uv.shape[1] / self.fs


def list_patient_files(raw_dir: Path) -> dict[str, Path]:
    """Map patient identifiers (``"01"`` ... ``"22"``) to their raw ``.h5`` file."""
    return {p.stem.split("_")[-1]: p for p in sorted(Path(raw_dir).glob("Data_Patient_*.h5"))}


def _decode_property(dataset: h5py.Dataset) -> str | float:
    """Decode a NIX property; strings saved one character per record are joined back."""
    values = dataset[()]["value"]
    if values.dtype == object:
        return "".join(v.decode("latin-1") if isinstance(v, bytes) else str(v) for v in values)
    return float(values[0])


def read_metadata(path: Path) -> dict[str, Any]:
    """Read patient-level clinical metadata from a raw file.

    Returns:
        Dictionary with keys ``age``, ``sex`` (``"F"``/``"M"``), ``etiology``,
        ``resection_area``, ``ilae`` (int), ``seizure_free`` (ILAE 1), ``follow_up_months``
        and ``electrode``.
    """
    with h5py.File(path, "r") as f:
        props = f["metadata/Patient/properties"]
        patient = {name: _decode_property(props[name]) for name in props}
        setup = f["metadata/General/sections/Recording setup/properties"]
        electrode = _decode_property(setup["Electrode type"])
    ilae = int(patient["Outcome ILAE"])
    return {
        "age": int(patient["Age"]),
        "sex": str(patient["Gender"])[0].upper(),
        "etiology": patient["Etiology"],
        "resection_area": patient["Resection area"],
        "ilae": ilae,
        "seizure_free": ilae == 1,
        "follow_up_months": int(patient["Follow-up period"]),
        "electrode": electrode,
    }


def _read_fr_markings(block: h5py.Group, ch_names: list[str]) -> pd.DataFrame:
    """Read FR tags of one block and convert the 1-based channel index to a channel name.

    Raises:
        ValueError: If the channel label embedded in a tag name (``FR_<k>_Channel_<i>_<label>``)
            does not match ``ch_names[i - 1]``. This guards the 1-based convention.
    """
    rows = []
    tags = block[FR_TAGS] if FR_TAGS in block else {}
    for key in tags:
        tag = tags[key]
        index_1based, onset_s = (float(v) for v in tag["position"][()])
        duration_s = float(tag["extent"][()][1])
        channel = ch_names[int(index_1based) - 1]
        label_in_name = str(tag.attrs["name"]).split("_")[-1]
        if label_in_name != channel:
            raise ValueError(
                f"Tag {tag.attrs['name']!r}: index {int(index_1based)} (1-based) points to "
                f"{channel!r}, but the tag name says {label_in_name!r}."
            )
        rows.append({"channel": channel, "onset_s": onset_s, "duration_s": duration_s})
    markings = pd.DataFrame(rows, columns=["channel", "onset_s", "duration_s"])
    return markings.sort_values(["onset_s", "channel"], ignore_index=True)


def read_nix_recording(path: Path, patient: str, session: str) -> Recording:
    """Read one recording block (signals, channel names, FR markings) from a raw NIX file.

    Args:
        path: Raw ``Data_Patient_XX.h5`` file.
        patient: Patient identifier stored in the returned object.
        session: ``"pre"`` or ``"post"``.
    """
    with h5py.File(path, "r") as f:
        block = f["data"][NIX_BLOCKS[session]]
        array = block[f"data_arrays/{SIGNAL_ARRAY}"]
        data_uv = array["data"][()]
        ch_names = [label.decode() for label in array["dimensions/1/labels"][()]]
        fs = 1.0 / float(array["dimensions/2"].attrs["sampling_interval"])
        markings = _read_fr_markings(block, ch_names)
    return Recording(patient, session, data_uv, round(fs, 6), ch_names, markings)


def recording_to_raw(
    rec: Recording,
    line_freq_hz: float = 50.0,
    highpass_hz: float = 1.0,
    lowpass_hz: float = 800.0,
) -> mne.io.RawArray:
    """Convert a :class:`Recording` to an MNE ``Raw`` object (volts, ECoG channels).

    FR markings become channel-specific ``Annotations`` with description ``"FR"``.
    ``highpass_hz`` / ``lowpass_hz`` record the *hardware* passband; no filtering is applied.
    """
    info = mne.create_info(rec.ch_names, sfreq=rec.fs, ch_types="ecog")
    raw = mne.io.RawArray(rec.data_uv * MICROVOLT, info, verbose=False)
    with raw.info._unlock():  # public setters do not exist for these documentation fields
        raw.info["line_freq"] = line_freq_hz
        raw.info["highpass"] = highpass_hz
        raw.info["lowpass"] = lowpass_hz
    m = rec.markings
    annotations = mne.Annotations(
        onset=m["onset_s"].to_numpy(),
        duration=m["duration_s"].to_numpy(),
        description=[FR_DESCRIPTION] * len(m),
        ch_names=[[ch] for ch in m["channel"]],
    )
    raw.set_annotations(annotations)
    return raw


def make_bids_path(
    bids_root: Path, patient: str, session: str, task: str = "intraop"
) -> mne_bids.BIDSPath:
    """BIDS path of one recording: ``sub-<patient>/ses-<pre|post>resection/ieeg``."""
    return mne_bids.BIDSPath(
        subject=patient,
        session=BIDS_SESSIONS[session],
        task=task,
        datatype="ieeg",
        root=Path(bids_root),
    )


def _update_ieeg_sidecar(
    bids_path: mne_bids.BIDSPath, highpass_hz: float, lowpass_hz: float
) -> None:
    """Fix the fields mne-bids cannot know: reference, hardware filters, real manufacturer."""
    sidecar = bids_path.copy().update(suffix="ieeg", extension=".json")
    mne_bids.update_sidecar_json(
        sidecar,
        {
            "iEEGReference": "bipolar (adjacent contacts on the same grid or strip)",
            "Manufacturer": "Natus Medical Incorporated",
            "ManufacturersModelName": "Nicolet C-Series",
            "HardwareFilters": {
                "HighpassFilter": {"CutoffFrequency": highpass_hz},
                "LowpassFilter": {"CutoffFrequency": lowpass_hz},
            },
            "SoftwareFilters": "n/a",
            "ElectricalStimulation": False,
        },
    )


def _write_events_with_channel(bids_path: mne_bids.BIDSPath, rec: Recording) -> None:
    """Rewrite ``events.tsv`` with an extra ``channel`` column (mne-bids drops it)."""
    m = rec.markings
    events = pd.DataFrame(
        {
            "onset": m["onset_s"].round(6),
            "duration": m["duration_s"].round(6),
            "trial_type": FR_DESCRIPTION,
            "channel": m["channel"],
            "sample": np.round(m["onset_s"] * rec.fs).astype(int),
        }
    )
    events_tsv = bids_path.copy().update(suffix="events", extension=".tsv").fpath
    events.to_csv(events_tsv, sep="\t", index=False, lineterminator="\n")
    description = {
        "onset": {"Description": "FR onset from the start of the recording", "Units": "s"},
        "duration": {"Description": "FR duration", "Units": "s"},
        "trial_type": {
            "Description": "Event type",
            "Levels": {
                "FR": "Fast ripple (250-500 Hz) found by an automatic detector and visually "
                "validated by the dataset authors (Boran et al., 2019)."
            },
        },
        "channel": {"Description": "Bipolar channel on which the FR was marked"},
        "sample": {"Description": "Onset as a 0-based sample index"},
    }
    events_json = events_tsv.with_suffix(".json")
    events_json.write_text(json.dumps(description, indent=2), encoding="utf-8", newline="\n")


def write_recording_to_bids(
    rec: Recording,
    bids_root: Path,
    task: str = "intraop",
    line_freq_hz: float = 50.0,
    highpass_hz: float = 1.0,
    lowpass_hz: float = 800.0,
) -> mne_bids.BIDSPath:
    """Write one recording to a BIDS dataset (BrainVision, float32) and return its path.

    Recordings with no FR still get an (empty) ``events.tsv`` so every recording has the same
    file set.
    """
    raw = recording_to_raw(rec, line_freq_hz, highpass_hz, lowpass_hz)
    bids_path = make_bids_path(bids_root, rec.patient, rec.session, task)
    with warnings.catch_warnings():
        # No electrode coordinates and no recording date exist for this dataset: expected.
        warnings.simplefilter("ignore", RuntimeWarning)
        mne_bids.write_raw_bids(
            raw,
            bids_path,
            format="BrainVision",
            allow_preload=True,
            overwrite=True,
            verbose=False,
        )
    _update_ieeg_sidecar(bids_path, highpass_hz, lowpass_hz)
    _write_events_with_channel(bids_path, rec)
    return bids_path


PARTICIPANT_COLUMNS = {
    "age": "Age at surgery (years)",
    "sex": "Sex (F or M)",
    "etiology": "Histopathology / etiology",
    "resection_area": "Resected area (free text from the authors; no channel-level labels)",
    "ilae": "ILAE surgical outcome class (1 = seizure free)",
    "seizure_free": "True if ILAE class 1",
    "follow_up_months": "Post-operative follow-up (months)",
    "electrode": "Electrode grid or strip type",
}


def write_participants(bids_root: Path, metadata: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Write ``participants.tsv`` / ``participants.json`` with the clinical metadata.

    Args:
        bids_root: BIDS root folder.
        metadata: Mapping ``patient -> read_metadata(...)`` output.
    """
    rows = [{"participant_id": f"sub-{pid}", **meta} for pid, meta in sorted(metadata.items())]
    table = pd.DataFrame(rows, columns=["participant_id", *PARTICIPANT_COLUMNS])
    table.to_csv(Path(bids_root) / "participants.tsv", sep="\t", index=False, lineterminator="\n")
    sidecar = {"participant_id": {"Description": "Unique participant identifier"}}
    sidecar |= {col: {"Description": desc} for col, desc in PARTICIPANT_COLUMNS.items()}
    (Path(bids_root) / "participants.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8", newline="\n"
    )
    return table


def write_dataset_description(bids_root: Path) -> None:
    """Write ``dataset_description.json`` with the original licence and citation."""
    mne_bids.make_dataset_description(
        path=bids_root,
        name="Intraoperative ECoG pre- and post-resection with fast ripple markings (BIDS)",
        dataset_type="raw",
        data_license="CC-BY-SA-4.0",
        how_to_acknowledge=(
            "Cite Boran E. et al., High-density ECoG improves the detection of high frequency "
            "oscillations that predict seizure outcome, Clin Neurophysiol 130(10):1882-1888, 2019."
        ),
        references_and_links=[
            "https://gin.g-node.org/USZ_NCH/Intraoperative_ECoG_HFO",
            "doi:10.1016/j.clinph.2019.07.008",
        ],
        generated_by=[{"Name": "epidetect", "Description": "scripts/01_convert_to_bids.py"}],
        overwrite=True,
        verbose=False,
    )


def read_bids_recording(
    bids_root: Path, patient: str, session: str, task: str = "intraop"
) -> mne.io.BaseRaw:
    """Read one BIDS recording, restoring channel-specific FR annotations.

    ``mne_bids.read_raw_bids`` ignores our ``channel`` column, so annotations are rebuilt from
    ``events.tsv``. Bad channels from ``channels.tsv`` are kept in ``raw.info["bads"]``.
    """
    bids_path = make_bids_path(bids_root, patient, session, task)
    with warnings.catch_warnings():
        # Missing electrode coordinates are expected for this dataset.
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    events_tsv = bids_path.copy().update(suffix="events", extension=".tsv").fpath
    events = pd.read_csv(events_tsv, sep="\t")
    fr = events[events["trial_type"] == FR_DESCRIPTION]
    raw.set_annotations(
        mne.Annotations(
            onset=fr["onset"].to_numpy(float),
            duration=fr["duration"].to_numpy(float),
            description=[FR_DESCRIPTION] * len(fr),
            ch_names=[[ch] for ch in fr["channel"]],
        )
    )
    return raw


def annotations_to_markings(raw: mne.io.BaseRaw, description: str = FR_DESCRIPTION) -> pd.DataFrame:
    """Channel-specific annotations of ``raw`` as a ``channel, onset_s, duration_s`` table."""
    rows = [
        {"channel": ch, "onset_s": float(a["onset"]), "duration_s": float(a["duration"])}
        for a in raw.annotations
        if a["description"] == description
        for ch in a["ch_names"]
    ]
    return pd.DataFrame(rows, columns=["channel", "onset_s", "duration_s"])
