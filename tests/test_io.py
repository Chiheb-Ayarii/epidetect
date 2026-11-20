"""Tests for epidetect.io: NIX reading, 1-based tag index, BIDS round trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import CH_NAMES, FR_TAGS, FS, write_fake_nix

from epidetect import io


def test_read_nix_recording_signals(fake_nix: tuple[Path, np.ndarray]) -> None:
    path, data = fake_nix
    rec = io.read_nix_recording(path, "01", "pre")
    assert rec.fs == FS
    assert rec.ch_names == CH_NAMES
    np.testing.assert_array_equal(rec.data_uv, data)
    assert rec.duration_s == pytest.approx(data.shape[1] / FS)


def test_fr_tags_use_one_based_channel_index(fake_nix: tuple[Path, np.ndarray]) -> None:
    path, _ = fake_nix
    rec = io.read_nix_recording(path, "01", "pre")
    expected = sorted((onset, CH_NAMES[idx - 1], dur) for idx, onset, dur in FR_TAGS)
    got = list(rec.markings[["onset_s", "channel", "duration_s"]].itertuples(index=False))
    assert [tuple(g) for g in got] == expected


def test_inconsistent_tag_name_raises(tmp_path: Path) -> None:
    path = tmp_path / "Data_Patient_02.h5"
    # Index 2 (1-based) is "02-03" but the tag name claims "03-04" (a 0-based reading).
    write_fake_nix(path, tags=[(2, 0.5, 0.01)], tag_labels=["03-04"])
    with pytest.raises(ValueError, match="1-based"):
        io.read_nix_recording(path, "02", "pre")


def test_post_block_without_markings(fake_nix: tuple[Path, np.ndarray]) -> None:
    path, _ = fake_nix
    rec = io.read_nix_recording(path, "01", "post")
    assert rec.markings.empty
    assert rec.ch_names == CH_NAMES[:2]


def test_read_metadata_joins_per_character_strings(fake_nix: tuple[Path, np.ndarray]) -> None:
    path, _ = fake_nix
    meta = io.read_metadata(path)
    assert meta["sex"] == "F"
    assert meta["etiology"] == "FCD 2a"
    assert meta["electrode"] == "1x6 strip"
    assert meta["ilae"] == 3 and meta["seizure_free"] is False


def test_recording_to_raw_units_and_annotations(fake_nix: tuple[Path, np.ndarray]) -> None:
    path, data = fake_nix
    raw = io.recording_to_raw(io.read_nix_recording(path, "01", "pre"))
    np.testing.assert_allclose(raw.get_data(), data * 1e-6)  # volts
    assert raw.info["line_freq"] == 50.0
    markings = io.annotations_to_markings(raw)
    assert sorted(markings["channel"]) == sorted(CH_NAMES[i - 1] for i, _, _ in FR_TAGS)


def test_bids_round_trip(fake_nix: tuple[Path, np.ndarray], tmp_path: Path) -> None:
    path, data = fake_nix
    bids_root = tmp_path / "bids"
    rec = io.read_nix_recording(path, "01", "pre")
    io.write_recording_to_bids(rec, bids_root)
    io.write_recording_to_bids(io.read_nix_recording(path, "01", "post"), bids_root)
    io.write_participants(bids_root, {"01": io.read_metadata(path)})
    io.write_dataset_description(bids_root)

    raw = io.read_bids_recording(bids_root, "01", "pre")
    # BrainVision stores float32: relative error ~1e-7.
    np.testing.assert_allclose(raw.get_data(), data * 1e-6, rtol=1e-6, atol=1e-12)
    back = io.annotations_to_markings(raw).sort_values("onset_s", ignore_index=True)
    pd.testing.assert_frame_equal(back, rec.markings, check_exact=False, atol=1e-6)

    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    assert participants.loc[0, "participant_id"] == "sub-01"
    assert participants.loc[0, "ilae"] == 3

    post = io.read_bids_recording(bids_root, "01", "post")
    assert len(post.annotations) == 0
