"""Leave-one-patient-out evaluation of the three detectors on the real fast-ripple markings.

Data: pre-resection recordings (excluded recordings skipped). Channels with repeated samples
(effective sampling rate below fs, e.g. P22 3-4) cannot carry fast ripples and are dropped.
Only the fast-ripple band is evaluated: the reference contains no ripple markings.

For each test patient p, two strategies choose the thresholds and train the ML forest
**without any data from p**:

* **A - real training patients**: energy k and wavelet z maximise the pooled F1 on the other
  patients; the forest is trained on their candidates; the probability threshold comes from
  out-of-fold predictions (GroupKFold over patients).
* **B - simulations**: the simulator is **recalibrated without p** (background spectrum, 1/f
  exponents, amplitude and spike-apex range recomputed from the other patients' channels and
  transients), training simulations are generated with it, and everything is tuned on them.

The test patient is scored channel by channel (detections never match across channels).
Patients without reference FRs only contribute false positives per minute. Summary statistics
are bootstrapped over patients. A sensitivity analysis scores only QC-clean channels.

Usage:
    python scripts/10_evaluate_real.py [--config configs/default.toml]   (~45 min, cached)

Outputs:
    results/real_fold_calibration.csv  simulator parameters of every fold vs all patients
    results/real_lopo_per_patient.csv  counts and metrics per analysis, strategy, detector, patient
    results/real_lopo_summary.csv      macro / micro metrics with 95 % bootstrap CIs
    figures/real_lopo.png
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

from epidetect import benchmark, evaluate, plotting, realdata
from epidetect.config import load_config
from epidetect.simulate import SimulationParams

BAND = "fast_ripple"
DETECTORS = {"energy": plotting.SERIES[0], "wavelet": plotting.SERIES[1], "ml": plotting.SERIES[2]}
STRATEGIES = {
    "A": "A: tuned on real training patients",
    "B": "B: tuned on recalibrated simulations",
}
RINGING_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


# --- Fold-wise simulator recalibration --------------------------------------------------------


def calibrated_values(cfg: dict, leave_out: str | None) -> dict:
    """Simulator parameters recomputed from all patients except ``leave_out``.

    Same rules as scripts/03_calibrate_simulator.py: medians of beta and crossover over
    channels whose floor is reached below the fit range, 2-40 Hz exponent and std from the QC,
    apex range minimising the log-quantile distance to the real ringing ratios. The simulated
    quantiles of each apex range come from results/spike_apex_grid.csv (they do not depend on
    any patient; they were computed with the all-patient background, a small approximation).
    """
    results = cfg["paths"]["results_dir"]
    fit_high = cfg["calibration"]["fit_range_hz"][1]
    fits = pd.read_csv(results / "background_fit.csv", dtype={"patient": str})
    qc = pd.read_csv(results / "qc_channels.csv", dtype={"patient": str})
    qc = qc[~qc.excluded_recording & ~qc.bad]
    ringing = pd.read_csv(results / "spike_ringing_real.csv", dtype={"patient": str})
    grid = pd.read_csv(results / "spike_apex_grid.csv")
    if leave_out is not None:
        fits, qc = fits[fits.patient != leave_out], qc[qc.patient != leave_out]
        ringing = ringing[ringing.patient != leave_out]
    reached = fits[fits.crossover_hz <= fit_high]
    real_q = np.quantile(ringing.ratio, RINGING_QUANTILES)
    sim_q = grid[[f"sim_q{int(q * 100)}" for q in RINGING_QUANTILES]].to_numpy()
    distance = np.sqrt(np.mean((np.log10(sim_q) - np.log10(real_q)) ** 2, axis=1))
    best = grid.iloc[int(np.argmin(distance))]
    return {
        "beta": float(reached.beta.median()),
        "crossover_hz": float(reached.crossover_hz.median()),
        "beta_low": float(qc.aperiodic_beta.median()),
        "std_uv": float(qc.std_uv.median()),
        "apex_low_ms": float(best.apex_low_ms),
        "apex_high_ms": float(best.apex_high_ms),
    }


def fold_params(base: SimulationParams, values: dict) -> SimulationParams:
    return base.replace(
        background={k: values[k] for k in ("beta", "crossover_hz", "beta_low", "std_uv")},
        spike={"apex_sigma_ms": (values["apex_low_ms"], values["apex_high_ms"])},
    )


def simulated_training(
    cfg: dict, params: SimulationParams, settings: benchmark.DetectorSettings, values: dict
) -> list[benchmark.TrainingUnit]:
    """Training simulations of one fold (cached by parameter values)."""
    bench = cfg["benchmark"]
    digest = hashlib.sha1(json.dumps(values, sort_keys=True).encode()).hexdigest()[:8]
    cache = benchmark.cache_dir(cfg, "real-fold-sims")
    units = []
    for i, snr in enumerate(bench["snr_levels_db"]):
        p = params.replace(duration_s=bench["sim_duration_s"], events={"snr_db": snr})
        n = bench["n_train_per_snr"]
        seeds = range(bench["train_seed_start"] + i * n, bench["train_seed_start"] + (i + 1) * n)
        units += [
            s.units[BAND]
            for s in benchmark.simulated_units(p, seeds, settings, cache, f"{digest}-snr{snr:g}")
        ]
    return units


# --- Scoring ----------------------------------------------------------------------------------


def score_patient(
    detections: dict[str, pd.DataFrame],
    unit: benchmark.TrainingUnit,
    minutes: dict[str, float],
    channels: list[str],
) -> dict[str, dict]:
    """Counts per detector restricted to ``channels``."""
    ref = unit.reference[unit.reference.channel.isin(channels)]
    out = {}
    for name, det in detections.items():
        counts = benchmark.count_by_channel(det[det.channel.isin(channels)], ref)
        out[name] = {**counts, "n_ref": len(ref), "minutes": sum(minutes[c] for c in channels)}
    return out


def per_patient_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = rows.apply(lambda r: evaluate.precision_recall_f1(r.tp, r.fp, r.fn), axis=1)
    rows = pd.concat([rows, pd.DataFrame(list(metrics), index=rows.index)], axis=1)
    rows["fp_per_min"] = rows.fp / rows.minutes
    rows.loc[rows.n_ref == 0, ["recall", "f1"]] = np.nan  # undefined without reference events
    return rows


def summarise(per_patient: pd.DataFrame, bench: dict) -> pd.DataFrame:
    statistics = {
        "macro_f1": evaluate.macro("f1"),
        "macro_recall": evaluate.macro("recall"),
        "macro_precision": evaluate.macro("precision"),
        "micro_f1": evaluate.pooled("f1"),
        "micro_recall": evaluate.pooled("recall"),
        "micro_precision": evaluate.pooled("precision"),
    }
    rows = []
    for keys, group in per_patient.groupby(["analysis", "strategy", "detector"], sort=False):
        with_ref = group[group.n_ref > 0]
        for name, statistic in statistics.items():
            est, low, high = evaluate.bootstrap_ci(
                with_ref,
                statistic,
                bench["n_bootstrap"],
                bench["ci_level"],
                bench["bootstrap_seed"],
            )
            rows.append(
                {
                    "analysis": keys[0],
                    "strategy": keys[1],
                    "detector": keys[2],
                    "metric": name,
                    "estimate": est,
                    "ci_low": low,
                    "ci_high": high,
                    "n_patients": len(with_ref),
                }
            )
        no_ref = group[group.n_ref == 0]
        est, low, high = evaluate.bootstrap_ci(
            no_ref,
            evaluate.ratio("fp", "minutes"),
            bench["n_bootstrap"],
            bench["ci_level"],
            bench["bootstrap_seed"],
        )
        rows.append(
            {
                "analysis": keys[0],
                "strategy": keys[1],
                "detector": keys[2],
                "metric": "fp_per_min_no_fr_patients",
                "estimate": est,
                "ci_low": low,
                "ci_high": high,
                "n_patients": len(no_ref),
            }
        )
    return pd.DataFrame(rows)


def plot(per_patient: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    main = per_patient[(per_patient.analysis == "all channels") & (per_patient.n_ref > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (strategy, title) in zip(axes, STRATEGIES.items(), strict=True):
        for k, (name, color) in enumerate(DETECTORS.items()):
            d = main[(main.strategy == strategy) & (main.detector == name)]
            x = k + rng.uniform(-0.15, 0.15, len(d))
            ax.scatter(x, d.f1, s=12 + 4 * np.sqrt(d.n_ref), color=color, alpha=0.55, lw=0)
            s = summary[
                (summary.analysis == "all channels")
                & (summary.strategy == strategy)
                & (summary.detector == name)
                & (summary.metric == "macro_f1")
            ].iloc[0]
            ax.errorbar(
                k + 0.32,
                s.estimate,
                yerr=[[s.estimate - s.ci_low], [s.ci_high - s.estimate]],
                fmt="D",
                color=plotting.INK,
                capsize=4,
                ms=6,
            )
            ax.text(k + 0.42, s.estimate, f"{s.estimate:.2f}", va="center", fontsize=8)
        ax.set_xticks(range(len(DETECTORS)), ["Energy", "Wavelet", "ML"])
        ax.set_xlim(-0.5, len(DETECTORS) - 0.2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, loc="left")
        ax.set_ylabel("F1 per test patient (fast ripples)")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=plotting.NEUTRAL,
            label="one patient (size ~ number of reference FRs)",
        ),
        plt.Line2D(
            [],
            [],
            marker="D",
            ls="",
            color=plotting.INK,
            label="macro F1, 95 % bootstrap CI over patients",
        ),
    ]
    fig.legend(handles=handles, loc="upper right", ncols=2, bbox_to_anchor=(1.0, 1.03))
    fig.suptitle("Leave-one-patient-out, real pre-resection data", x=0.01, ha="left", y=1.03)
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    bench = cfg["benchmark"]
    results = cfg["paths"]["results_dir"]
    full = benchmark.settings_from_config(cfg)
    settings = dataclasses.replace(full, bands={BAND: full.bands[BAND]})
    base = SimulationParams.from_config(cfg["simulation"])

    channels = realdata.channel_table(cfg)
    units, minutes = realdata.load_real_units(cfg, settings, channels)
    for patient, unit in units.items():
        print(f"patient {patient}: {len(minutes[patient])} channels, {len(unit.reference)} FRs")

    calibration = [{"left_out": "none (all patients)", **calibrated_values(cfg, None)}]
    rows = []
    for test in sorted(units):
        train = [units[p] for p in sorted(units) if p != test]
        values = calibrated_values(cfg, test)
        calibration.append({"left_out": test, **values})
        tuned = {
            "A": benchmark.tune_band(train, BAND, settings),
            "B": benchmark.tune_band(
                simulated_training(cfg, fold_params(base, values), settings, values), BAND, settings
            ),
        }
        clean = channels[(channels.patient == test) & ~channels.bad].channel.tolist()
        analyses = {
            "all channels": channels[channels.patient == test].channel.tolist(),
            "QC-clean channels": clean,
        }
        for strategy, t in tuned.items():
            detections = benchmark.detect_tuned(t, units[test])
            for analysis, chans in analyses.items():
                for name, counts in score_patient(
                    detections, units[test], minutes[test], chans
                ).items():
                    rows.append(
                        {
                            "analysis": analysis,
                            "strategy": strategy,
                            "detector": name,
                            "patient": test,
                            "energy_k": t.energy_k,
                            "wavelet_z": t.wavelet_z,
                            "ml_threshold": t.ml_threshold,
                            **counts,
                        }
                    )
        print(
            f"fold {test} done (A: k={tuned['A'].energy_k}, z={tuned['A'].wavelet_z}, "
            f"p={tuned['A'].ml_threshold}; B: k={tuned['B'].energy_k}, "
            f"z={tuned['B'].wavelet_z}, p={tuned['B'].ml_threshold})"
        )

    pd.DataFrame(calibration).round(4).to_csv(
        results / "real_fold_calibration.csv", index=False, lineterminator="\n"
    )
    per_patient = per_patient_metrics(pd.DataFrame(rows))
    per_patient.round(4).to_csv(
        results / "real_lopo_per_patient.csv", index=False, lineterminator="\n"
    )
    summary = summarise(per_patient, bench)
    summary.round(4).to_csv(results / "real_lopo_summary.csv", index=False, lineterminator="\n")
    plot(per_patient, summary, cfg["paths"]["figures_dir"] / "real_lopo.png")
    print(
        summary.pivot_table(
            index=["analysis", "strategy", "detector"], columns="metric", values="estimate"
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
