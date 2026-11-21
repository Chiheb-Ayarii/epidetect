"""Clinical analysis: do post-resection fast ripples predict the surgical outcome?

Patient level (the analysis supported by this dataset): one score per patient (rate of
post-resection fast ripples) against a binary outcome (not seizure-free = ILAE > 1).

* ROC curve and AUC, computed from ranks: AUC = P(score of a random positive > score of a
  random negative), ties counting 1/2. This is the Mann-Whitney U statistic divided by
  n_pos * n_neg.
* 95 % CI by **stratified** bootstrap: positives and negatives are resampled separately, so
  that every replicate keeps both classes (with 8 positives, a plain bootstrap would sometimes
  draw none).
* p-value by **permutation** of the outcome labels: the probability, under "no association",
  of an AUC at least as far from 0.5 as observed (two-sided).

Channel level (ready, not run: no channel-level resection labels exist for this dataset):
``channel_level_auc`` gives, for each patient, the AUC of channel FR rates against
"resected / not resected" labels.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve from ranks (ties count 1/2); NaN without both classes."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)  # average ranks for ties
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False-positive rate and true-positive rate for every threshold (from +inf down)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    thresholds = np.concatenate([[np.inf], np.unique(scores)[::-1]])
    tpr = np.array([(scores[labels] >= t).mean() for t in thresholds])
    fpr = np.array([(scores[~labels] >= t).mean() for t in thresholds])
    return fpr, tpr


def stratified_bootstrap(
    scores: np.ndarray,
    labels: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float] = roc_auc,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Estimate and percentile CI, resampling positives and negatives separately."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        rows = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values.append(statistic(scores[rows], labels[rows]))
    alpha = (1 - level) / 2
    low, high = np.nanquantile(values, [alpha, 1 - alpha])
    return statistic(scores, labels), float(low), float(high)


def permutation_pvalue(
    scores: np.ndarray, labels: np.ndarray, n_perm: int = 10000, seed: int = 0
) -> float:
    """Two-sided permutation p-value of the AUC (distance from 0.5), with the +1 correction."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    observed = abs(roc_auc(scores, labels) - 0.5)
    rng = np.random.default_rng(seed)
    hits = sum(
        abs(roc_auc(scores, rng.permutation(labels)) - 0.5) >= observed - 1e-12
        for _ in range(n_perm)
    )
    return float((hits + 1) / (n_perm + 1))


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of a proportion (better than the normal approximation for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    from scipy.stats import norm

    z = norm.ppf(0.5 + level / 2)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def rule_metrics(
    predicted: np.ndarray, labels: np.ndarray, level: float = 0.95
) -> dict[str, float]:
    """Sensitivity, specificity, PPV and NPV of a binary rule, with Wilson intervals."""
    predicted, labels = np.asarray(predicted, bool), np.asarray(labels, bool)
    out: dict[str, float] = {}
    for name, hits, n in [
        ("sensitivity", (predicted & labels).sum(), labels.sum()),
        ("specificity", (~predicted & ~labels).sum(), (~labels).sum()),
        ("ppv", (predicted & labels).sum(), predicted.sum()),
        ("npv", (~predicted & ~labels).sum(), (~predicted).sum()),
    ]:
        low, high = wilson_interval(int(hits), int(n), level)
        out |= {
            name: hits / n if n else float("nan"),
            f"{name}_low": low,
            f"{name}_high": high,
            f"{name}_n": int(n),
        }
    return out


def channel_level_auc(
    table: pd.DataFrame, score: str = "fr_rate", label: str = "resected"
) -> pd.DataFrame:
    """Per-patient AUC of channel scores against channel resection labels.

    Args:
        table: One row per channel with columns ``patient``, ``score`` and ``label`` (bool).

    Returns:
        One row per patient: ``n_channels``, ``n_resected`` and ``auc`` (NaN if a patient has
        only resected or only non-resected channels).
    """
    rows = []
    for patient, group in table.groupby("patient"):
        rows.append(
            {
                "patient": patient,
                "n_channels": len(group),
                "n_resected": int(group[label].sum()),
                "auc": roc_auc(group[score].to_numpy(), group[label].to_numpy(bool)),
            }
        )
    return pd.DataFrame(rows)
