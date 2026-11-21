"""Strategy C: tune the detectors on a hybrid simulation built without the test patient.

For each test patient p (leave-one-patient-out, as in scripts/10_evaluate_real.py):

1. the hybrid calibration (FR rate, share on spikes, frequency/duration pairs, in-band SNR)
   is pooled from the reference markings of the **other** patients only;
2. synthetic FRs are inserted into real background windows (no marked FR) of the **other**
   patients only;
3. energy k and wavelet z are tuned on these hybrid windows, the ML forest is trained on them
   and its probability threshold chosen out-of-fold (GroupKFold over patients);
4. the tuned detectors are applied to p, scored channel by channel.

Strategy C is then compared with A (tuned on real training patients) and B (tuned on the
fully synthetic simulator), from results/real_lopo_per_patient.csv: thresholds, macro / micro
F1, false positives per minute, and paired per-patient F1 differences (bootstrap over patients).

Usage:
    python scripts/12_evaluate_hybrid.py [--config configs/default.toml]   (~40 min, cached)

Outputs:
    results/hybrid_fold_calibration.csv  calibration of every fold (and of all patients)
    results/hybrid_lopo_per_patient.csv  per (analysis, detector, patient), strategy C
    results/strategy_comparison.csv      A vs B vs C: metrics and paired differences, 95 % CIs
    results/strategy_thresholds.csv      thresholds chosen in every fold, per strategy
    figures/strategy_comparison.png
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epidetect import benchmark, evaluate, hybrid, io, plotting, realdata
from epidetect.config import load_config

BAND = realdata.FR_BAND_NAME
STRATEGY_LABELS = {
    "A": "A: real training patients",
    "B": "B: synthetic simulator",
    "C": "C: hybrid simulation",
}
STRATEGY_COLORS = {"A": plotting.SERIES[0], "B": plotting.NEUTRAL, "C": plotting.SERIES[1]}
DETECTOR_LABELS = {"energy": "Energy", "wavelet": "Wavelet", "ml": "ML"}


def prepare_patient(
    cfg: dict, patient: str, channels: pd.DataFrame, hp: hybrid.HybridParams
) -> dict:
    """Marking statistics, analysed minutes and background windows of one patient."""
    raw = io.read_bids_recording(cfg["paths"]["bids_root"], patient, "pre", cfg["bids"]["task"])
    fs = raw.info["sfreq"]
    markings = io.annotations_to_markings(raw)
    clean = channels[(channels.patient == patient) & ~channels.bad].channel.tolist()
    stats, minutes, data = [], 0.0, {}
    for channel in clean:
        x = raw.get_data(picks=[channel])[0] / io.MICROVOLT
        data[channel] = x
        minutes += (len(x) / fs - 2 * hp.edge_margin_s) / 60
        m = markings[markings.channel == channel]
        if len(m):
            stats.append(hybrid.marking_statistics(x, fs, m, hp))
    rng = np.random.default_rng([hp.seed, int(patient)])
    windows = []
    for _ in range(hp.windows_per_patient * 20):  # pick a channel, then a free window in it
        if len(windows) == hp.windows_per_patient or not clean:
            break
        channel = clean[int(rng.integers(len(clean)))]
        m = markings[markings.channel == channel]
        intervals = (
            np.column_stack([m.onset_s, m.onset_s + m.duration_s]) if len(m) else np.empty((0, 2))
        )
        taken = [w["start"] for w in windows if w["channel"] == channel]
        for start in hybrid.background_window_starts(len(data[channel]), fs, intervals, hp, rng, 1):
            if all(abs(start - s) >= hp.window_s * fs for s in taken):
                n = int(hp.window_s * fs)
                windows.append(
                    {
                        "channel": channel,
                        "start": start,
                        "x": data[channel][start : start + n].astype(np.float32),
                    }
                )
    stats = (
        pd.concat(stats, ignore_index=True)
        if stats
        else pd.DataFrame(columns=["freq_hz", "duration_s", "snr_db", "on_spike"])
    )
    return {"fs": fs, "stats": stats, "minutes": minutes, "windows": windows}


def fold_units(
    cfg: dict,
    prepared: dict,
    train: list[str],
    calibration: hybrid.HybridCalibration,
    hp: hybrid.HybridParams,
    settings: benchmark.DetectorSettings,
) -> list[benchmark.TrainingUnit]:
    """Hybrid training units (one per training patient), cached by calibration and window."""
    key = json.dumps(
        {"calibration": calibration.summary(), "hybrid": dataclasses.asdict(hp)},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    cache = benchmark.cache_dir(cfg, "hybrid-folds")
    units = []
    for patient in train:
        info = prepared[patient]

        def compute(patient: str = patient, info: dict = info) -> benchmark.TrainingUnit:
            per_window = {}
            for k, w in enumerate(info["windows"]):
                rng = np.random.default_rng([hp.seed, int(patient), k])
                signal, truth = hybrid.insert_fast_ripples(
                    w["x"].astype(float), info["fs"], calibration, hp, rng
                )
                per_window[f"{w['channel']}#{k}"] = benchmark.analyse_signal(
                    signal, info["fs"], truth, settings
                )[BAND]
            return benchmark.merge_channels(per_window, patient)

        if info["windows"]:
            units.append(benchmark.load_or_compute(cache / f"{digest}-{patient}.pkl", compute))
    return units


def score(
    detections: dict, unit: benchmark.TrainingUnit, minutes: dict, channels: list[str]
) -> dict:
    ref = unit.reference[unit.reference.channel.isin(channels)]
    return {
        name: {
            **benchmark.count_by_channel(det[det.channel.isin(channels)], ref),
            "n_ref": len(ref),
            "minutes": sum(minutes[c] for c in channels),
        }
        for name, det in detections.items()
    }


def compare(per_patient: pd.DataFrame, bench: dict) -> pd.DataFrame:
    """Macro / micro metrics per strategy and paired per-patient F1 differences."""
    args = (bench["n_bootstrap"], bench["ci_level"], bench["bootstrap_seed"])
    rows = []
    for (analysis, detector), group in per_patient.groupby(["analysis", "detector"]):
        with_ref = group[group.n_ref > 0]
        for strategy, g in with_ref.groupby("strategy"):
            for name, stat in [
                ("macro_f1", evaluate.macro("f1")),
                ("micro_f1", evaluate.pooled("f1")),
                ("macro_precision", evaluate.macro("precision")),
                ("macro_recall", evaluate.macro("recall")),
            ]:
                est, low, high = evaluate.bootstrap_ci(g, stat, *args)
                rows.append(
                    {
                        "analysis": analysis,
                        "detector": detector,
                        "comparison": strategy,
                        "metric": name,
                        "estimate": est,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            no_ref = group[(group.n_ref == 0) & (group.strategy == strategy)]
            est, low, high = evaluate.bootstrap_ci(no_ref, evaluate.ratio("fp", "minutes"), *args)
            rows.append(
                {
                    "analysis": analysis,
                    "detector": detector,
                    "comparison": strategy,
                    "metric": "fp_per_min_no_fr_patients",
                    "estimate": est,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        wide = with_ref.pivot_table(index="patient", columns="strategy", values="f1")
        for a, b in [("C", "B"), ("C", "A")]:
            diff = pd.DataFrame({"diff": wide[a] - wide[b]})
            est, low, high = evaluate.bootstrap_ci(diff, evaluate.macro("diff"), *args)
            rows.append(
                {
                    "analysis": analysis,
                    "detector": detector,
                    "comparison": f"{a} - {b}",
                    "metric": "paired_f1_difference",
                    "estimate": est,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def plot(comparison: pd.DataFrame, thresholds: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    main = comparison[(comparison.analysis == "all channels") & (comparison.metric == "macro_f1")]
    for k, detector in enumerate(DETECTOR_LABELS):
        for j, strategy in enumerate(STRATEGY_LABELS):
            r = main[(main.detector == detector) & (main.comparison == strategy)].iloc[0]
            x = k + (j - 1) * 0.25
            axes[0].bar(
                x,
                r.estimate,
                width=0.22,
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy] if k == 0 else None,
            )
            axes[0].errorbar(
                x,
                r.estimate,
                yerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]],
                color=plotting.INK,
                capsize=3,
                lw=1,
            )
    axes[0].set_xticks(range(3), DETECTOR_LABELS.values())
    axes[0].set_ylabel("Macro F1 on test patients (fast ripples)")
    axes[0].set_title("Leave-one-patient-out: macro F1, 95 % CI", loc="left")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="x", visible=False)
    for ax, column, title in [
        (axes[1], "energy_k", "Energy threshold k chosen per fold"),
        (axes[2], "wavelet_z", "Wavelet threshold z chosen per fold"),
    ]:
        for j, strategy in enumerate(STRATEGY_LABELS):
            values = thresholds.loc[thresholds.strategy == strategy, column].to_numpy()
            jitter = np.random.default_rng(j).uniform(-0.12, 0.12, len(values))
            ax.scatter(j + jitter, values, color=STRATEGY_COLORS[strategy], s=18, alpha=0.7, lw=0)
        ax.set_xticks(range(3), ["A", "B", "C"])
        ax.set_title(title, loc="left")
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    bench, results = cfg["benchmark"], cfg["paths"]["results_dir"]
    hp = hybrid.HybridParams.from_config(cfg["hybrid"])
    full = benchmark.settings_from_config(cfg)
    settings = dataclasses.replace(full, bands={BAND: full.bands[BAND]})

    channels = realdata.channel_table(cfg)
    units, minutes = realdata.load_real_units(cfg, settings, channels)
    prep_cache = benchmark.cache_dir(cfg, "hybrid-prep")
    prep_key = hashlib.sha1(json.dumps(dataclasses.asdict(hp), default=str).encode()).hexdigest()[
        :8
    ]
    prepared = {
        p: benchmark.load_or_compute(
            prep_cache / f"{prep_key}-{p}.pkl", lambda p=p: prepare_patient(cfg, p, channels, hp)
        )
        for p in sorted(units)
    }

    def calibrate(patients: list[str]) -> hybrid.HybridCalibration:
        stats = pd.concat([prepared[p]["stats"] for p in patients], ignore_index=True)
        return hybrid.HybridCalibration.from_statistics(
            stats, sum(prepared[p]["minutes"] for p in patients)
        )

    calibration_rows = [{"left_out": "none (all patients)", **calibrate(sorted(units)).summary()}]
    rows, thresholds = [], []
    for test in sorted(units):
        train = [p for p in sorted(units) if p != test]
        calibration = calibrate(train)
        calibration_rows.append({"left_out": test, **calibration.summary()})
        tuned = benchmark.tune_band(
            fold_units(cfg, prepared, train, calibration, hp, settings), BAND, settings
        )
        thresholds.append(
            {
                "strategy": "C",
                "patient": test,
                "energy_k": tuned.energy_k,
                "wavelet_z": tuned.wavelet_z,
                "ml_threshold": tuned.ml_threshold,
            }
        )
        detections = benchmark.detect_tuned(tuned, units[test])
        clean = channels[(channels.patient == test) & ~channels.bad].channel.tolist()
        for analysis, chans in {
            "all channels": list(minutes[test]),
            "QC-clean channels": clean,
        }.items():
            for name, counts in score(detections, units[test], minutes[test], chans).items():
                rows.append(
                    {
                        "analysis": analysis,
                        "strategy": "C",
                        "detector": name,
                        "patient": test,
                        **counts,
                    }
                )
        print(
            f"fold {test}: k={tuned.energy_k}, z={tuned.wavelet_z}, p={tuned.ml_threshold}, "
            f"{len(calibration.freq_hz)} training markings"
        )

    pd.DataFrame(calibration_rows).round(4).to_csv(
        results / "hybrid_fold_calibration.csv", index=False, lineterminator="\n"
    )
    hybrid_rows = pd.DataFrame(rows)
    metrics = hybrid_rows.apply(lambda r: evaluate.precision_recall_f1(r.tp, r.fp, r.fn), axis=1)
    hybrid_rows = pd.concat(
        [hybrid_rows, pd.DataFrame(list(metrics), index=hybrid_rows.index)], axis=1
    )
    hybrid_rows["fp_per_min"] = hybrid_rows.fp / hybrid_rows.minutes
    hybrid_rows.loc[hybrid_rows.n_ref == 0, ["recall", "f1"]] = np.nan
    hybrid_rows.round(4).to_csv(
        results / "hybrid_lopo_per_patient.csv", index=False, lineterminator="\n"
    )

    ab = pd.read_csv(results / "real_lopo_per_patient.csv", dtype={"patient": str})
    columns = [
        "analysis",
        "strategy",
        "detector",
        "patient",
        "tp",
        "fp",
        "fn",
        "n_ref",
        "minutes",
        "precision",
        "recall",
        "f1",
        "fp_per_min",
    ]
    per_patient = pd.concat([ab[columns], hybrid_rows[columns]], ignore_index=True)
    comparison = compare(per_patient, bench)
    comparison.round(4).to_csv(
        results / "strategy_comparison.csv", index=False, lineterminator="\n"
    )
    ab_thresholds = ab[(ab.analysis == "all channels") & (ab.detector == "energy")][
        ["strategy", "patient", "energy_k", "wavelet_z", "ml_threshold"]
    ]
    all_thresholds = pd.concat([ab_thresholds, pd.DataFrame(thresholds)], ignore_index=True)
    all_thresholds.to_csv(results / "strategy_thresholds.csv", index=False, lineterminator="\n")
    plot(comparison, all_thresholds, cfg["paths"]["figures_dir"] / "strategy_comparison.png")

    print(all_thresholds.groupby("strategy")[["energy_k", "wavelet_z", "ml_threshold"]].median())
    main_rows = comparison[comparison.analysis == "all channels"]
    print(
        main_rows.pivot_table(index=["detector", "comparison"], columns="metric", values="estimate")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
