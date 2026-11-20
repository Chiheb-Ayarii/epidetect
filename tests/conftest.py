"""Shared fixtures: a tiny file with the same NIX layout as the real dataset.

The fake file contains random noise only. It exists to test the *reader*, never to produce
results, and it is written to pytest's temporary folder (never to data/).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

FS = 2000.0
CH_NAMES = ["01-02", "02-03", "03-04", "05-06"]
N_SAMPLES = 4000  # 2 s
# (1-based channel index, onset s, duration s) -- same convention as the real tags.
FR_TAGS = [(2, 0.50, 0.015), (4, 1.20, 0.020), (2, 1.60, 0.012)]

PROPERTY_DTYPE = np.dtype(
    [
        ("value", h5py.string_dtype()),
        ("uncertainty", "<f8"),
        ("reference", h5py.string_dtype()),
        ("filename", h5py.string_dtype()),
        ("encoder", h5py.string_dtype()),
        ("checksum", h5py.string_dtype()),
    ]
)
NUMBER_DTYPE = np.dtype(
    [
        (n, "<f8") if n in ("value", "uncertainty") else (n, h5py.string_dtype())
        for n in PROPERTY_DTYPE.names
    ]
)


def _write_string_property(group: h5py.Group, name: str, value: str, per_char: bool) -> None:
    """NIX string property; ``per_char`` mimics the one-record-per-letter quirk."""
    parts = list(value) if per_char else [value]
    records = np.array([(p, 0.0, "", "", "", "") for p in parts], dtype=PROPERTY_DTYPE)
    group.create_dataset(name, data=records)


def _write_number_property(group: h5py.Group, name: str, value: float) -> None:
    records = np.array([(value, 0.0, "", "", "", "")], dtype=NUMBER_DTYPE)
    group.create_dataset(name, data=records)


def write_fake_nix(
    path: Path, tags: list[tuple[int, float, float]] = FR_TAGS, tag_labels: list[str] | None = None
) -> np.ndarray:
    """Write a minimal NIX-like file and return the pre-resection data (uV)."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 20.0, (len(CH_NAMES), N_SAMPLES))
    with h5py.File(path, "w") as f:
        props = f.create_group("metadata/Patient/properties")
        _write_number_property(props, "Age", 15.0)
        _write_string_property(props, "Gender", "Female", per_char=True)
        _write_string_property(props, "Etiology", "FCD 2a", per_char=True)
        _write_string_property(props, "Resection area", "L medial parietal", per_char=False)
        _write_number_property(props, "Follow-up period", 24.0)
        _write_number_property(props, "Outcome ILAE", 3.0)
        setup = f.create_group("metadata/General/sections/Recording setup/properties")
        _write_string_property(setup, "Electrode type", "1x6 strip", per_char=True)
        for block_name, block_data, block_tags in [
            ("Data_Pre_Resection_Bipolar_Montage", data, tags),
            ("Data_Post_Resection_Bipolar_Montage", data[:2], []),
        ]:
            arr = f.create_group(f"data/{block_name}/data_arrays/ECoG_Bipolar_Channels")
            arr.create_dataset("data", data=block_data)
            names = CH_NAMES[: block_data.shape[0]]
            arr.create_dataset(
                "dimensions/1/labels", data=np.array(names, dtype=object), dtype=h5py.string_dtype()
            )
            arr["dimensions/2"] = h5py.Empty("f")
            arr["dimensions/2"].attrs["sampling_interval"] = 1.0 / FS
            group = f.create_group(f"data/{block_name}/groups/FR markings/tags")
            for k, (idx, onset, dur) in enumerate(block_tags):
                label = tag_labels[k] if tag_labels else CH_NAMES[idx - 1]
                tag = group.create_group(f"uuid-{k}")
                tag.attrs["name"] = f"FR_{k + 1}_Channel_{idx:02d}_{label}"
                tag.create_dataset("position", data=np.array([idx, onset]))
                tag.create_dataset("extent", data=np.array([1.0, dur]))
    return data


@pytest.fixture
def fake_nix(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Path of a fake ``Data_Patient_01.h5`` and its pre-resection data."""
    path = tmp_path / "Data_Patient_01.h5"
    data = write_fake_nix(path)
    return path, data
