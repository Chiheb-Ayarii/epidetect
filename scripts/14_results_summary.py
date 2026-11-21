"""Write the README results tables from the result files (no number is typed by hand).

The tables are inserted between the markers ``<!-- RESULTS:START -->`` and
``<!-- RESULTS:END -->`` of README.md, and also saved to ``results/summary_tables.md``.

Usage:
    python scripts/14_results_summary.py [--config configs/default.toml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epidetect.config import REPO_ROOT, load_config

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
DETECTORS = {"energy": "Energy (RMS)", "wavelet": "Wavelet (Morlet)", "ml": "ML (random forest)"}
SUMMARY_SNR_DB = 10.0


def ci(value: float, low: float, high: float, digits: int = 2) -> str:
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def simulation_table(results: Path) -> str:
    m = pd.read_csv(results / "sim_eval_metrics.csv")
    m = m[m.snr_db == SUMMARY_SNR_DB].set_index(["band", "detector", "metric"])
    lines = [
        f"**Simulations, SNR {SUMMARY_SNR_DB:g} dB** (`results/sim_eval_metrics.csv`; "
        "95 % bootstrap CI over 10 test channels of 120 s)",
        "",
        "| Band | Detector | F1 | Precision | Recall, isolated HFOs | Recall, HFOs on a spike "
        "| Isolated spikes detected |",
        "|---|---|---|---|---|---|---|",
    ]
    for band, label in [("fast_ripple", "Fast ripples"), ("ripple", "Ripples")]:
        for det, name in DETECTORS.items():

            def row(metric: str, b: str = band, d: str = det) -> pd.Series:
                return m.loc[(b, d, metric)]

            f1 = row("f1")
            lines.append(
                f"| {label} | {name} | {ci(f1.estimate, f1.ci_low, f1.ci_high)} "
                f"| {row('precision').estimate:.2f} | {row('recall_isolated').estimate:.2f} "
                f"| {row('recall_on_spike').estimate:.2f} "
                f"| {100 * row('spike_false_alarm_rate').estimate:.0f} % |"
            )
    return "\n".join(lines)


def real_table(results: Path) -> str:
    c = pd.read_csv(results / "strategy_comparison.csv")
    c = c[c.analysis == "all channels"].set_index(["detector", "comparison", "metric"])
    th = (
        pd.read_csv(results / "strategy_thresholds.csv")
        .groupby("strategy")
        .median(numeric_only=True)
    )
    names = {
        "A": "A: tuned on real training patients",
        "B": "B: tuned on synthetic simulator",
        "C": "C: tuned on hybrid simulation",
    }
    lines = [
        "**Real data, fast ripples, leave-one-patient-out** (`results/strategy_comparison.csv`; "
        "16 patients with reference FRs, 95 % bootstrap CI over patients)",
        "",
        "| Strategy | Detector | Macro F1 | Micro F1 | FP / min (3 patients without FR) "
        "| Median threshold |",
        "|---|---|---|---|---|---|",
    ]
    for strategy, label in names.items():
        for det, name in DETECTORS.items():
            macro = c.loc[(det, strategy, "macro_f1")]
            micro = c.loc[(det, strategy, "micro_f1")]
            fp = c.loc[(det, strategy, "fp_per_min_no_fr_patients")]
            threshold = {
                "energy": f"k = {th.loc[strategy, 'energy_k']:g}",
                "wavelet": f"z = {th.loc[strategy, 'wavelet_z']:g}",
                "ml": f"p = {th.loc[strategy, 'ml_threshold']:g}",
            }[det]
            lines.append(
                f"| {label} | {name} | {ci(macro.estimate, macro.ci_low, macro.ci_high)} "
                f"| {micro.estimate:.2f} | {fp.estimate:.2f} | {threshold} |"
            )
    return "\n".join(lines)


def clinical_table(results: Path) -> str:
    a = pd.read_csv(results / "clinical_auc.csv")
    a = a[a.analysis == "all channels"]
    rule = pd.read_csv(results / "clinical_rule.csv").iloc[0]
    names = {
        "reference": "Reference markings",
        **{k: v + " detector" for k, v in DETECTORS.items()},
    }
    lines = [
        "**Clinical, patient level** (`results/clinical_auc.csv`; post-resection FR rate vs "
        f"not seizure-free; {int(a.n_not_seizure_free.iloc[0])} vs "
        f"{int(a.n_seizure_free.iloc[0])} patients; stratified bootstrap CI)",
        "",
        "| Score | AUC [95 % CI] | Permutation p |",
        "|---|---|---|",
    ]
    for row in a.itertuples():
        lines.append(
            f"| {names[row.score]} | {ci(row.auc, row.ci_low, row.ci_high)} "
            f"| {row.p_permutation:.3f} |"
        )
    lines += [
        "",
        f'Rule "at least one reference FR after resection": sensitivity '
        f"{int(round(rule.sensitivity * rule.sensitivity_n))}/{int(rule.sensitivity_n)}, "
        f"specificity {int(round(rule.specificity * rule.specificity_n))}/"
        f"{int(rule.specificity_n)} (`results/clinical_rule.csv`).",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = cfg["paths"]["results_dir"]
    tables = "\n\n".join([simulation_table(results), real_table(results), clinical_table(results)])
    (results / "summary_tables.md").write_text(tables + "\n", encoding="utf-8", newline="\n")
    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        readme.write_text(f"{head}{START}\n{tables}\n{END}{tail}", encoding="utf-8", newline="\n")
        print("README results tables updated")
    print(tables)


if __name__ == "__main__":
    main()
