"""Threshold tuning shared by the simulation and real-data benchmarks.

Rules that prevent leakage:

* every threshold is chosen on **training units only** (simulated channels or patients);
* the ML probability threshold is chosen on **out-of-fold** predictions: the training units are
  split into groups (GroupKFold), each group is predicted by a forest trained on the others,
  so the threshold is never chosen on predictions of a model that saw the same data.

Grid choice: the value with the highest pooled F1; among ties, the first value of the grid.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from epidetect import evaluate, simulate
from epidetect.detectors.base import bands_from_config
from epidetect.detectors.energy import EnergyParams, detect_energy
from epidetect.detectors.ml import MLDetector, MLParams, label_candidates
from epidetect.detectors.wavelet import WaveletParams, detect_wavelet
from epidetect.features import FEATURE_NAMES
from epidetect.simulate import SimulationParams


@dataclass(frozen=True)
class TrainingUnit:
    """Detector outputs and reference of one training unit, in one band.

    Attributes:
        unit: Identifier (simulation seed or patient).
        reference: Reference events of the band (``onset_s``, ``offset_s``, optional
            ``channel``).
        energy: Energy detections for every grid value ``k`` (dict ``k -> table``).
        wavelet: Wavelet detections at the lowest grid threshold (filtered later by ``score``).
        candidates: ML candidate table with features and a ``label`` column.
    """

    unit: str
    reference: pd.DataFrame
    energy: dict[float, pd.DataFrame]
    wavelet: pd.DataFrame
    candidates: pd.DataFrame


def count_by_channel(detections: pd.DataFrame, reference: pd.DataFrame) -> dict[str, int]:
    """TP / FP / FN with matching done channel by channel (if a ``channel`` column exists)."""
    if "channel" not in reference.columns and "channel" not in detections.columns:
        return evaluate.count_matches(detections, reference)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    channels = set(detections.get("channel", pd.Series(dtype=str))) | set(
        reference.get("channel", pd.Series(dtype=str))
    )
    for channel in channels:
        counts = evaluate.count_matches(
            detections[detections["channel"] == channel], reference[reference["channel"] == channel]
        )
        for key in totals:
            totals[key] += counts[key]
    return totals


def best_on_grid(grid: Sequence[float], f1_of: Callable[[float], float]) -> tuple[float, float]:
    """Grid value with the highest F1 (first one among ties) and that F1."""
    scores = [f1_of(value) for value in grid]
    best = int(np.nanargmax(scores))
    return float(grid[best]), float(scores[best])


def pooled_f1(counts: Sequence[dict[str, int]]) -> float:
    """Micro-averaged F1 over several units."""
    tp, fp, fn = (sum(c[k] for c in counts) for k in ("tp", "fp", "fn"))
    return evaluate.precision_recall_f1(tp, fp, fn)["f1"]


def tune_energy(units: Sequence[TrainingUnit], grid: Sequence[float]) -> tuple[float, float]:
    """Energy threshold k maximising the pooled F1 over the training units."""
    return best_on_grid(
        grid, lambda k: pooled_f1([count_by_channel(u.energy[k], u.reference) for u in units])
    )


def tune_wavelet(units: Sequence[TrainingUnit], grid: Sequence[float]) -> tuple[float, float]:
    """Wavelet z threshold maximising the pooled F1 (detections filtered by ``score``)."""

    def f1_of(z: float) -> float:
        return pooled_f1(
            [count_by_channel(u.wavelet[u.wavelet["score"] >= z], u.reference) for u in units]
        )

    return best_on_grid(grid, f1_of)


def out_of_fold_probabilities(
    table: pd.DataFrame, groups: np.ndarray, make_detector: Callable[[], MLDetector], n_splits: int
) -> np.ndarray:
    """Probability of each candidate predicted by a forest trained on the other groups."""
    probabilities = np.full(len(table), np.nan)
    n_splits = min(n_splits, len(np.unique(groups)))
    for train_rows, test_rows in GroupKFold(n_splits=n_splits).split(table, groups=groups):
        detector = make_detector().fit_table(table.iloc[train_rows])
        scored = detector.predict_table(table.iloc[test_rows])
        probabilities[test_rows] = scored["score"].to_numpy(float)
    return probabilities


def tune_ml_threshold(
    table: pd.DataFrame,
    probabilities: np.ndarray,
    n_reference: int,
    grid: Sequence[float],
) -> tuple[float, float]:
    """Probability threshold maximising the pooled F1 of out-of-fold predictions.

    Args:
        table: Candidates with a ``label`` column (1 = matches a reference event).
        probabilities: Out-of-fold probabilities (NaN = band not learnable in that fold,
            treated as 0).
        n_reference: Number of reference events, *including* those never captured by any
            candidate: they are false negatives whatever the threshold.
    """
    labels = table["label"].to_numpy(int)
    p = np.nan_to_num(probabilities, nan=0.0)

    def f1_of(threshold: float) -> float:
        predicted = p >= threshold
        tp = int((predicted & (labels == 1)).sum())
        fp = int((predicted & (labels == 0)).sum())
        return evaluate.precision_recall_f1(tp, fp, n_reference - tp)["f1"]

    return best_on_grid(grid, f1_of)


# --- Running all detectors on one signal, tuning, applying ------------------------------------


@dataclass(frozen=True)
class DetectorSettings:
    """Everything needed to run and tune the three detectors."""

    bands: dict[str, tuple[float, float]]
    energy: EnergyParams
    wavelet: WaveletParams
    ml: MLParams
    edge_margin_s: float
    energy_grid: tuple[float, ...]
    wavelet_grid: tuple[float, ...]
    probability_grid: tuple[float, ...]
    inner_folds: int = 5
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)

    def ml_detector(self) -> MLDetector:
        return MLDetector(
            self.bands, self.ml, self.energy, self.wavelet, self.edge_margin_s, self.feature_names
        )


def analyse_signal(
    x_uv: np.ndarray, fs: float, reference: pd.DataFrame, settings: DetectorSettings
) -> dict[str, dict[str, Any]]:
    """Run everything needed for tuning and testing on one channel, split by band.

    Returns, for each band: ``energy`` (dict k -> detections), ``wavelet`` (detections at the
    lowest grid threshold), ``candidates`` (ML candidate table with ``label``) and
    ``reference`` (events of the band). ``reference`` needs ``kind``, ``onset_s``, ``offset_s``.
    """
    s = settings
    energy = {
        k: detect_energy(x_uv, fs, s.bands, replace(s.energy, threshold_k=k), s.edge_margin_s)
        for k in s.energy_grid
    }
    wavelet = detect_wavelet(
        x_uv, fs, s.bands, replace(s.wavelet, z_threshold=min(s.wavelet_grid)), s.edge_margin_s
    )
    candidates = s.ml_detector().table(x_uv, fs)
    candidates["label"] = label_candidates(candidates, reference)
    out = {}
    for band in s.bands:
        out[band] = {
            "energy": {k: d[d["band"] == band].reset_index(drop=True) for k, d in energy.items()},
            "wavelet": wavelet[wavelet["band"] == band].reset_index(drop=True),
            "candidates": candidates[candidates["band"] == band].reset_index(drop=True),
            "reference": reference[reference["kind"] == band].reset_index(drop=True),
        }
    return out


def merge_channels(per_channel: dict[str, dict[str, Any]], unit: str) -> TrainingUnit:
    """Combine the analyses of several channels (one band) into one unit with a channel column."""

    def tagged(name: str, table: pd.DataFrame) -> pd.DataFrame:
        return table.assign(channel=name)

    names = list(per_channel)
    grid = per_channel[names[0]]["energy"].keys() if names else []
    return TrainingUnit(
        unit=unit,
        reference=pd.concat(
            [tagged(n, per_channel[n]["reference"]) for n in names], ignore_index=True
        ),
        energy={
            k: pd.concat([tagged(n, per_channel[n]["energy"][k]) for n in names], ignore_index=True)
            for k in grid
        },
        wavelet=pd.concat([tagged(n, per_channel[n]["wavelet"]) for n in names], ignore_index=True),
        candidates=pd.concat(
            [tagged(n, per_channel[n]["candidates"]) for n in names], ignore_index=True
        ),
    )


@dataclass
class TunedDetectors:
    """Thresholds chosen on training units, and the trained forest (one band)."""

    band: str
    energy_k: float
    wavelet_z: float
    ml_threshold: float
    ml: MLDetector
    train_f1: dict[str, float]


def tune_band(
    units: Sequence[TrainingUnit], band: str, settings: DetectorSettings
) -> TunedDetectors:
    """Tune the three detectors of one band on training units only."""
    energy_k, f1_energy = tune_energy(units, settings.energy_grid)
    wavelet_z, f1_wavelet = tune_wavelet(units, settings.wavelet_grid)
    table = pd.concat([u.candidates.assign(unit=u.unit) for u in units], ignore_index=True)
    groups = table["unit"].to_numpy()
    probabilities = out_of_fold_probabilities(
        table, groups, settings.ml_detector, settings.inner_folds
    )
    n_reference = sum(len(u.reference) for u in units)
    ml_threshold, f1_ml = tune_ml_threshold(
        table, probabilities, n_reference, settings.probability_grid
    )
    ml = settings.ml_detector().fit_table(table)
    return TunedDetectors(
        band,
        energy_k,
        wavelet_z,
        ml_threshold,
        ml,
        {"energy": f1_energy, "wavelet": f1_wavelet, "ml": f1_ml},
    )


def detect_tuned(tuned: TunedDetectors, unit: TrainingUnit) -> dict[str, pd.DataFrame]:
    """Detections of the three tuned detectors on a (test) unit, for the tuned band."""
    candidates = unit.candidates
    if len(candidates) and tuned.band in tuned.ml.models:
        scored = tuned.ml.predict_table(candidates)
        ml = scored[scored["score"] >= tuned.ml_threshold]
    else:
        ml = candidates.iloc[0:0]
    return {
        "energy": unit.energy[tuned.energy_k],
        "wavelet": unit.wavelet[unit.wavelet["score"] >= tuned.wavelet_z],
        "ml": ml,
    }


# --- Configuration and cache -------------------------------------------------------------------


def settings_from_config(
    cfg: dict[str, Any], feature_names: tuple[str, ...] | None = None
) -> DetectorSettings:
    """DetectorSettings from ``[detectors]``, ``[features]`` and ``[benchmark]``."""
    det, bench = cfg["detectors"], cfg["benchmark"]
    return DetectorSettings(
        bands=bands_from_config(det),
        energy=EnergyParams.from_config(det["energy"]),
        wavelet=WaveletParams.from_config(det["wavelet"]),
        ml=MLParams.from_config(det["ml"], cfg["features"]),
        edge_margin_s=det["edge_margin_s"],
        energy_grid=tuple(bench["energy_k_grid"]),
        wavelet_grid=tuple(bench["wavelet_z_grid"]),
        probability_grid=tuple(bench["probability_grid"]),
        inner_folds=bench["inner_folds"],
        feature_names=feature_names or tuple(FEATURE_NAMES),
    )


def cache_dir(cfg: dict[str, Any], name: str) -> Path:
    """``data/derived/<name>-<hash>``: the hash covers every config section that can change a
    cached analysis, so a parameter change never reuses stale results."""
    relevant = {k: cfg[k] for k in ("simulation", "detectors", "features", "benchmark", "data")}
    digest = hashlib.sha1(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[
        :10
    ]
    path = cfg["paths"]["raw_dir"].parent / "derived" / f"{name}-{digest}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_or_compute(path: Path, compute: Callable[[], Any]) -> Any:
    """Return the pickled result at ``path``, computing and saving it first if needed."""
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - our own cache, never downloaded data
    result = compute()
    with path.open("wb") as f:
        pickle.dump(result, f)
    return result


# --- Simulation units and metric summaries ----------------------------------------------------


@dataclass(frozen=True)
class SimulatedUnit:
    """One simulated channel: its ground truth, analysed duration and per-band units."""

    seed: int
    truth: pd.DataFrame
    minutes: float
    units: dict[str, TrainingUnit]


def simulated_units(
    params: SimulationParams,
    seeds: Sequence[int],
    settings: DetectorSettings,
    cache: Path,
    tag: str,
) -> list[SimulatedUnit]:
    """Simulate and analyse one channel per seed (cached under ``cache/<tag>-<seed>.pkl``)."""
    out = []
    for seed in seeds:

        def compute(seed: int = seed) -> SimulatedUnit:
            sim = simulate.simulate_from_seed(params, seed)
            analysed = analyse_signal(sim.signal_uv, sim.fs, sim.events, settings)
            minutes = (params.duration_s - 2 * settings.edge_margin_s) / 60
            units = {b: merge_channels({"sim": analysed[b]}, str(seed)) for b in settings.bands}
            return SimulatedUnit(seed, sim.events, minutes, units)

        out.append(load_or_compute(cache / f"{tag}-{seed}.pkl", compute))
    return out


METRICS: dict[str, Callable[[pd.DataFrame], float]] = {
    "precision": evaluate.pooled("precision"),
    "recall": evaluate.pooled("recall"),
    "f1": evaluate.pooled("f1"),
    "recall_isolated": evaluate.ratio("tp_isolated", "n_isolated"),
    "recall_on_spike": evaluate.ratio("tp_on_spike", "n_on_spike"),
    "spike_false_alarm_rate": evaluate.ratio("spikes_hit", "n_spikes"),
    "fp_per_min": evaluate.ratio("fp", "minutes"),
}


def summarise(
    counts: pd.DataFrame,
    by: list[str],
    metrics: dict[str, Callable[[pd.DataFrame], float]],
    n_boot: int,
    level: float,
    seed: int,
) -> pd.DataFrame:
    """Estimate and bootstrap CI of each metric, for each group of ``by`` (rows = units)."""
    rows = []
    for keys, group in counts.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for name, statistic in metrics.items():
            estimate, low, high = evaluate.bootstrap_ci(group, statistic, n_boot, level, seed)
            rows.append(
                {
                    **dict(zip(by, keys, strict=True)),
                    "metric": name,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_units": len(group),
                }
            )
    return pd.DataFrame(rows)
