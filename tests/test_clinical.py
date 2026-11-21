"""Tests for the clinical statistics (AUC, bootstrap, permutation, rules, channel level)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from epidetect import clinical


def test_auc_perfect_inverse_and_ties() -> None:
    labels = np.array([0, 0, 0, 1, 1], dtype=bool)
    assert clinical.roc_auc(np.array([1, 2, 3, 4, 5]), labels) == 1.0
    assert clinical.roc_auc(np.array([5, 4, 3, 2, 1]), labels) == 0.0
    assert clinical.roc_auc(np.zeros(5), labels) == 0.5  # all tied: chance
    assert np.isnan(clinical.roc_auc(np.arange(3), np.ones(3, bool)))


def test_auc_equals_mann_whitney_u() -> None:
    rng = np.random.default_rng(0)
    scores = np.round(rng.normal(size=30), 1)  # rounding creates ties
    labels = rng.random(30) < 0.4
    u = mannwhitneyu(scores[labels], scores[~labels]).statistic
    assert clinical.roc_auc(scores, labels) == pytest.approx(u / (labels.sum() * (~labels).sum()))


def test_roc_curve_endpoints_and_area() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1], dtype=bool)
    fpr, tpr = clinical.roc_curve(scores, labels)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1)
    assert np.trapezoid(tpr, fpr) == pytest.approx(clinical.roc_auc(scores, labels))


def test_stratified_bootstrap_keeps_both_classes() -> None:
    scores = np.arange(10.0)
    labels = np.array([0] * 8 + [1] * 2, dtype=bool)
    est, low, high = clinical.stratified_bootstrap(scores, labels, n_boot=300)
    assert est == 1.0 and low == high == 1.0  # every replicate has both classes, never NaN


def test_permutation_pvalue_small_for_separation_large_for_noise() -> None:
    labels = np.array([0] * 10 + [1] * 8, dtype=bool)
    assert clinical.permutation_pvalue(np.arange(18.0), labels, n_perm=2000) < 0.01
    noise = np.random.default_rng(1).normal(size=18)
    assert clinical.permutation_pvalue(noise, labels, n_perm=2000) > 0.05


def test_rule_metrics_and_wilson() -> None:
    m = clinical.rule_metrics(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 0, 1, 1], bool))
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == 0.5
    low, high = clinical.wilson_interval(0, 8)
    assert low == pytest.approx(0.0, abs=1e-12) and 0.3 < high < 0.4


def test_channel_level_auc_per_patient() -> None:
    table = pd.DataFrame(
        {
            "patient": ["01"] * 4 + ["02"] * 3,
            "fr_rate": [5.0, 4.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "resected": [True, True, False, False, True, True, True],
        }
    )
    out = clinical.channel_level_auc(table).set_index("patient")
    assert out.loc["01", "auc"] == 1.0 and out.loc["01", "n_resected"] == 2
    assert np.isnan(out.loc["02", "auc"])  # only resected channels: undefined
