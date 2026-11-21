"""Patient-level clinical analysis: do post-resection fast ripples predict the surgical outcome?

Outcome: not seizure-free (ILAE > 1) = positive class; seizure-free (ILAE 1) = negative.
Score of a patient: rate of fast ripples in the post-resection recording (events per minute,
summed over channels and divided by the total channel-minutes), computed with:
    * the reference markings (automatic detector + visual validation, by the dataset authors);
    * each of our three detectors. For patient p, thresholds and forest come from the fold that
      **excluded p**: tuned on the pre-resection FR markings of the other patients (strategy A).
      The surgical outcome is never used to tune anything.

Statistics: ROC AUC, 95 % CI by stratified bootstrap (positives and negatives resampled
separately), two-sided permutation p-value; for the reference, the rule "at least one
post-resection FR" with sensitivity / specificity (Wilson intervals). A sensitivity analysis
uses QC-clean channels only.

Channel level: if ``data/resection_labels.csv`` exists (columns ``patient, channel,
resected``), the per-patient AUC of pre-resection channel FR rates against the resection
labels is computed. No such labels exist for this dataset, so this part is skipped.

Usage:
    python scripts/13_clinical.py [--config configs/default.toml]   (~30 min, cached)

Outputs:
    results/clinical_patients.csv   one row per patient: outcome, post-resection FR counts/rates
    results/clinical_auc.csv        AUC, 95 % CI, permutation p-value per score and analysis
    results/clinical_rule.csv       "any reference FR after resection" rule
    figures/clinical_roc.png
    (results/clinical_channel_auc.csv only if channel-level resection labels are provided)
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epidetect import benchmark, clinical, plotting, realdata
from epidetect.config import load_config

BAND = realdata.FR_BAND_NAME
SCORES = {
    "reference": "Reference markings",
    "energy": "Energy detector",
    "wavelet": "Wavelet detector",
    "ml": "ML detector",
}
SCORE_COLORS = {
    "reference": plotting.INK,
    "energy": plotting.SERIES[0],
    "wavelet": plotting.SERIES[1],
    "ml": plotting.SERIES[2],
}
N_PERMUTATIONS = 10000
RESECTION_LABELS = "resection_labels.csv"


def tuned_without(cfg: dict, patient: str, pre_units: dict, settings) -> benchmark.TunedDetectors:
    """Strategy-A detectors tuned on the pre-resection data of every other patient (cached)."""
    train = [u for p, u in sorted(pre_units.items()) if p != patient]
    key = "all" if patient not in pre_units else patient  # same training set for absent patients
    cache = benchmark.cache_dir(cfg, "tuned-A")
    return benchmark.load_or_compute(
        cache / f"without-{key}.pkl", lambda: benchmark.tune_band(train, BAND, settings)
    )


def patient_rows(
    post_units: dict,
    post_minutes: dict,
    tuned: dict,
    channels: pd.DataFrame,
    participants: pd.DataFrame,
    clean_only: bool,
) -> pd.DataFrame:
    rows = []
    for patient, unit in sorted(post_units.items()):
        chans = list(post_minutes[patient])
        if clean_only:
            chans = channels[(channels.patient == patient) & ~channels.bad].channel.tolist()
        minutes = sum(post_minutes[patient][c] for c in chans)
        counts = {"reference": int(unit.reference.channel.isin(chans).sum())}
        for name, det in benchmark.detect_tuned(tuned[patient], unit).items():
            counts[name] = int(det.channel.isin(chans).sum())
        info = participants.loc[patient]
        rows.append(
            {
                "patient": patient,
                "ilae": int(info.ilae),
                "not_seizure_free": int(info.ilae) > 1,
                "n_channels": len(chans),
                "channel_minutes": minutes,
                **{f"n_{k}": v for k, v in counts.items()},
                **{
                    f"rate_{k}": (v / minutes if minutes > 0 else np.nan) for k, v in counts.items()
                },
            }
        )
    return pd.DataFrame(rows)


def auc_table(table: pd.DataFrame, analysis: str, bench: dict) -> pd.DataFrame:
    labels = table.not_seizure_free.to_numpy(bool)
    rows = []
    for name in SCORES:
        scores = table[f"rate_{name}"].to_numpy(float)
        est, low, high = clinical.stratified_bootstrap(
            scores,
            labels,
            n_boot=bench["n_bootstrap"],
            level=bench["ci_level"],
            seed=bench["bootstrap_seed"],
        )
        rows.append(
            {
                "analysis": analysis,
                "score": name,
                "auc": est,
                "ci_low": low,
                "ci_high": high,
                "p_permutation": clinical.permutation_pvalue(scores, labels, N_PERMUTATIONS),
                "n_not_seizure_free": int(labels.sum()),
                "n_seizure_free": int((~labels).sum()),
            }
        )
    return pd.DataFrame(rows)


def channel_level(cfg: dict, pre_units: dict, pre_minutes: dict) -> None:
    """Per-patient AUC of pre-resection channel FR rates vs resection labels, if available."""
    path = cfg["paths"]["raw_dir"].parent / RESECTION_LABELS
    if not path.exists():
        print(
            f"Channel-level analysis skipped: {path} not found (no channel-level resection "
            "labels exist for this dataset)."
        )
        return
    labels = pd.read_csv(path, dtype={"patient": str})
    rows = []
    for patient, unit in pre_units.items():
        for channel, minutes in pre_minutes[patient].items():
            n = int((unit.reference.channel == channel).sum())
            rows.append({"patient": patient, "channel": channel, "fr_rate": n / minutes})
    table = pd.DataFrame(rows).merge(labels, on=["patient", "channel"], how="inner")
    out = clinical.channel_level_auc(table, score="fr_rate", label="resected")
    out.to_csv(
        cfg["paths"]["results_dir"] / "clinical_channel_auc.csv", index=False, lineterminator="\n"
    )
    print(out.to_string(index=False))


def plot(table: pd.DataFrame, aucs: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    labels = table.not_seizure_free.to_numpy(bool)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=plotting.GRID, ls="--", lw=1)
    for name, label in SCORES.items():
        fpr, tpr = clinical.roc_curve(table[f"rate_{name}"].to_numpy(float), labels)
        r = aucs[(aucs.analysis == "all channels") & (aucs.score == name)].iloc[0]
        ax.step(
            fpr,
            tpr,
            where="post",
            color=SCORE_COLORS[name],
            lw=2 if name == "reference" else 1.5,
            label=(
                f"{label}: AUC {r.auc:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}], "
                f"p={r.p_permutation:.3f}"
            ),
        )
    ax.set_xlabel("1 - specificity (seizure-free patients flagged)")
    ax.set_ylabel("Sensitivity (not seizure-free patients flagged)")
    ax.set_title("Post-resection FR rate predicting a poor outcome", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    rng = np.random.default_rng(0)
    for k, name in enumerate(SCORES):
        for outcome, offset, marker in [(False, -0.17, "o"), (True, 0.17, "^")]:
            values = table.loc[table.not_seizure_free == outcome, f"rate_{name}"].to_numpy(float)
            ax.scatter(
                k + offset + rng.uniform(-0.06, 0.06, len(values)),
                values,
                s=26,
                color=SCORE_COLORS[name],
                marker=marker,
                alpha=0.75,
                lw=0,
            )
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(range(len(SCORES)), [s.replace(" ", "\n", 1) for s in SCORES.values()])
    ax.set_ylabel("Post-resection FR rate (per channel-minute)")
    ax.set_title("Each patient: o seizure-free (ILAE 1), ^ not seizure-free (ILAE > 1)", loc="left")
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    bench, results = cfg["benchmark"], cfg["paths"]["results_dir"]
    full = benchmark.settings_from_config(cfg)
    settings = dataclasses.replace(full, bands={BAND: full.bands[BAND]})
    participants = pd.read_csv(cfg["paths"]["bids_root"] / "participants.tsv", sep="\t")
    participants.index = participants.participant_id.str.replace("sub-", "", regex=False)

    pre_channels = realdata.channel_table(cfg, "pre")
    pre_units, pre_minutes = realdata.load_real_units(cfg, settings, pre_channels, "pre")
    post_channels = realdata.channel_table(cfg, "post")
    post_units, post_minutes = realdata.load_real_units(cfg, settings, post_channels, "post")
    tuned = {p: tuned_without(cfg, p, pre_units, settings) for p in sorted(post_units)}

    tables, aucs = [], []
    for analysis, clean in [("all channels", False), ("QC-clean channels", True)]:
        table = patient_rows(post_units, post_minutes, tuned, post_channels, participants, clean)
        table.insert(0, "analysis", analysis)
        tables.append(table)
        aucs.append(auc_table(table, analysis, bench))
    patients, aucs = pd.concat(tables, ignore_index=True), pd.concat(aucs, ignore_index=True)
    patients.round(5).to_csv(results / "clinical_patients.csv", index=False, lineterminator="\n")
    aucs.round(4).to_csv(results / "clinical_auc.csv", index=False, lineterminator="\n")

    main_table = patients[patients.analysis == "all channels"]
    rule = clinical.rule_metrics(
        main_table.n_reference.to_numpy() > 0,
        main_table.not_seizure_free.to_numpy(bool),
        bench["ci_level"],
    )
    pd.DataFrame([rule]).round(4).to_csv(
        results / "clinical_rule.csv", index=False, lineterminator="\n"
    )
    plot(main_table, aucs, cfg["paths"]["figures_dir"] / "clinical_roc.png")
    channel_level(cfg, pre_units, pre_minutes)

    print(
        main_table[
            ["patient", "ilae", "n_channels", "n_reference", "n_energy", "n_wavelet", "n_ml"]
        ].to_string(index=False)
    )
    print(aucs.round(3).to_string(index=False))
    print({k: round(v, 3) for k, v in rule.items() if not k.endswith("_n")})


if __name__ == "__main__":
    main()
