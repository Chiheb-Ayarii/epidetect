"""Machine-learning HFO detector: permissive candidates + random-forest classification.

1. **Candidates**: the energy detector with a *low* threshold and a relaxed oscillation
   criterion catches almost every HFO, together with many spikes and noise bursts.
2. **Features**: ``epidetect.features`` (oscillation, spike indicators, time-frequency).
3. **Classifier**: one random forest per band, ``class_weight="balanced_subsample"`` because
   candidates are dominated by negatives (class imbalance). A candidate is a positive example
   if it matches (one-to-one interval overlap, ``epidetect.evaluate``) a reference event of
   its band.

Feature extraction (slow) is separated from training (fast) so that cross-validation, e.g.
leave-one-patient-out in step 4, re-trains without recomputing features. Training data must
never contain the recording, channel or patient being tested.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from epidetect import evaluate
from epidetect.detectors.base import DETECTION_COLUMNS, empty_detections
from epidetect.detectors.energy import EnergyParams, detect_energy
from epidetect.detectors.wavelet import WaveletParams
from epidetect.features import FEATURE_NAMES, FeatureParams, extract_features


@dataclass(frozen=True)
class MLParams:
    """ML detector parameters (defaults mirror ``[detectors.ml]`` in the config)."""

    candidate_threshold_k: float = 3.0
    candidate_min_peaks: int = 4
    n_estimators: int = 300
    min_samples_leaf: int = 3
    probability_threshold: float = 0.5
    random_state: int = 0
    features: FeatureParams = field(default_factory=FeatureParams)

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], feature_cfg: dict[str, Any] | None = None
    ) -> MLParams:
        return cls(**cfg, features=FeatureParams.from_config(feature_cfg or {}))


def find_candidates(
    x_uv: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    params: MLParams,
    energy: EnergyParams | None = None,
    edge_margin_s: float = 1.0,
) -> pd.DataFrame:
    """Permissive energy detections used as candidates."""
    permissive = dataclasses.replace(
        energy or EnergyParams(),
        threshold_k=params.candidate_threshold_k,
        min_peaks=params.candidate_min_peaks,
    )
    return detect_energy(x_uv, fs, bands, permissive, edge_margin_s)


def candidate_table(
    x_uv: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    params: MLParams,
    energy: EnergyParams | None = None,
    wavelet: WaveletParams | None = None,
    edge_margin_s: float = 1.0,
) -> pd.DataFrame:
    """Candidates of one channel with their features (``DETECTION_COLUMNS + FEATURE_NAMES``)."""
    candidates = find_candidates(x_uv, fs, bands, params, energy, edge_margin_s)
    features = extract_features(
        x_uv, fs, candidates, bands, params.features, wavelet, edge_margin_s
    )
    # freq_hz is both a detection column and a feature (same zero-crossing estimate): keep one.
    duplicated = [c for c in features.columns if c in candidates.columns]
    return pd.concat(
        [
            candidates.reset_index(drop=True),
            features.drop(columns=duplicated).reset_index(drop=True),
        ],
        axis=1,
    )


def label_candidates(table: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    """1 if a candidate matches a reference event of its band (one-to-one), else 0.

    ``reference`` needs ``onset_s``, ``offset_s`` and ``kind`` (``ripple``/``fast_ripple``).
    """
    labels = np.zeros(len(table), dtype=int)
    for band in table["band"].unique():
        rows = np.flatnonzero(table["band"].to_numpy() == band)
        det_match, _ = evaluate.match_events(table.iloc[rows], reference[reference["kind"] == band])
        labels[rows[det_match >= 0]] = 1
    return labels


class MLDetector:
    """Random-forest classification of energy candidates, one model per band."""

    def __init__(
        self,
        bands: dict[str, tuple[float, float]],
        params: MLParams | None = None,
        energy: EnergyParams | None = None,
        wavelet: WaveletParams | None = None,
        edge_margin_s: float = 1.0,
    ) -> None:
        self.bands = bands
        self.params = params or MLParams()
        self.energy = energy
        self.wavelet = wavelet
        self.edge_margin_s = edge_margin_s
        self.models: dict[str, RandomForestClassifier] = {}

    def table(self, x_uv: np.ndarray, fs: float) -> pd.DataFrame:
        """Candidates and features of one channel."""
        return candidate_table(
            x_uv, fs, self.bands, self.params, self.energy, self.wavelet, self.edge_margin_s
        )

    def training_table(
        self, examples: Iterable[tuple[np.ndarray, float, pd.DataFrame]]
    ) -> pd.DataFrame:
        """Candidate tables of several labelled signals, with a ``label`` column."""
        tables = []
        for k, (x_uv, fs, reference) in enumerate(examples):
            table = self.table(x_uv, fs)
            table["label"] = label_candidates(table, reference)
            table["example"] = k
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    def fit_table(self, table: pd.DataFrame) -> MLDetector:
        """Train one forest per band present with both classes in ``table``."""
        self.models = {}
        for band, group in table.groupby("band"):
            if group["label"].nunique() < 2:
                continue  # cannot learn a band without positive (or negative) examples
            model = RandomForestClassifier(
                n_estimators=self.params.n_estimators,
                min_samples_leaf=self.params.min_samples_leaf,
                class_weight="balanced_subsample",
                random_state=self.params.random_state,
                n_jobs=-1,
            )
            model.fit(_feature_matrix(group), group["label"].to_numpy())
            self.models[str(band)] = model
        return self

    def fit(self, examples: Iterable[tuple[np.ndarray, float, pd.DataFrame]]) -> MLDetector:
        """Extract candidates and features from labelled signals, then train."""
        return self.fit_table(self.training_table(examples))

    def predict_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Add a ``score`` (probability of being an HFO) to a candidate table.

        Bands without a trained model get ``score = NaN``.
        """
        scored = table.copy()
        scored["score"] = np.nan
        for band, model in self.models.items():
            rows = scored["band"] == band
            if rows.any():
                scored.loc[rows, "score"] = model.predict_proba(_feature_matrix(scored[rows]))[:, 1]
        return scored

    def detections_from_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Keep candidates whose probability reaches ``probability_threshold``."""
        scored = self.predict_table(table)
        kept = scored[scored["score"] >= self.params.probability_threshold]
        return (
            kept[DETECTION_COLUMNS].reset_index(drop=True) if not kept.empty else empty_detections()
        )

    def detect(self, x_uv: np.ndarray, fs: float) -> pd.DataFrame:
        """Detections of one channel (standard detector output, ``score`` = probability)."""
        return self.detections_from_table(self.table(x_uv, fs))

    def feature_importances(self) -> pd.DataFrame:
        """Impurity-based importances, one column per band."""
        return pd.DataFrame(
            {band: m.feature_importances_ for band, m in self.models.items()}, index=FEATURE_NAMES
        )


def _feature_matrix(table: pd.DataFrame) -> np.ndarray:
    """Feature columns as a float matrix; missing time-frequency values become -1 (a value
    that cannot occur for a ratio of z-scores near a peak, so the forest can isolate it)."""
    return table[FEATURE_NAMES].astype(float).fillna(-1.0).to_numpy()
