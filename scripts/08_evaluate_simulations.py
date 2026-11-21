"""Benchmark the three detectors on simulations: per class, per SNR and per event sub-group.

Protocol (``[benchmark]`` in the config):
    * training channels: ``n_train_per_snr`` simulations at each SNR level (seeds from
      ``train_seed_start``); test channels: ``n_test_per_snr`` per level (seeds from
      ``test_seed_start``). Seed ranges are disjoint: no simulation is used for both.
    * per band, the energy threshold k and the wavelet threshold z are chosen on the training
      channels (pooled F1); the ML forest is trained on them and its probability threshold is
      chosen on out-of-fold predictions (GroupKFold over training channels).
    * on every test channel, each tuned detector is scored per band: precision, recall, F1,
      recall on isolated HFOs, recall on HFOs riding on a spike, false alarms on isolated
      spikes, false positives per minute.
    * 95 % confidence intervals: percentile bootstrap over the test channels of each SNR.

Usage:
    python scripts/08_evaluate_simulations.py [--config configs/default.toml]

Outputs:
    results/sim_tuned_thresholds.csv   thresholds chosen on the training simulations
    results/sim_eval_counts.csv        one row per (test channel, detector, band)
    results/sim_eval_metrics.csv       estimates and 95 % CIs per (detector, band, SNR, metric)
    figures/sim_benchmark_snr.png      recall / precision / spike false alarms versus SNR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epidetect import benchmark, evaluate, plotting
from epidetect.config import load_config
from epidetect.simulate import SimulationParams

DETECTORS = {"energy": plotting.SERIES[0], "wavelet": plotting.SERIES[1], "ml": plotting.SERIES[2]}
LABELS = {"energy": "Energy (RMS)", "wavelet": "Wavelet", "ml": "ML (random forest)"}
PANELS = [
    ("recall_isolated", "Recall, isolated HFOs"),
    ("recall_on_spike", "Recall, HFOs on a spike"),
    ("precision", "Precision"),
    ("spike_false_alarm_rate", "Isolated spikes falsely detected"),
]


def build_sets(
    cfg: dict, settings: benchmark.DetectorSettings, variant: str, base: SimulationParams
) -> tuple[dict, dict]:
    """Training and test simulated channels, per SNR level (cached)."""
    bench = cfg["benchmark"]
    cache = benchmark.cache_dir(cfg, "sim")
    train, test = {}, {}
    for i, snr in enumerate(bench["snr_levels_db"]):
        params = base.replace(duration_s=bench["sim_duration_s"], events={"snr_db": snr})
        n_tr, n_te = bench["n_train_per_snr"], bench["n_test_per_snr"]
        train_seeds = range(
            bench["train_seed_start"] + i * n_tr, bench["train_seed_start"] + (i + 1) * n_tr
        )
        test_seeds = range(
            bench["test_seed_start"] + i * n_te, bench["test_seed_start"] + (i + 1) * n_te
        )
        train[snr] = benchmark.simulated_units(
            params, train_seeds, settings, cache, f"{variant}-train-snr{snr:g}"
        )
        test[snr] = benchmark.simulated_units(
            params, test_seeds, settings, cache, f"{variant}-test-snr{snr:g}"
        )
    return train, test


def tune(train: dict, settings: benchmark.DetectorSettings) -> dict[str, benchmark.TunedDetectors]:
    """Tune each band on all training channels (all SNR levels together)."""
    channels = [unit for units in train.values() for unit in units]
    return {
        band: benchmark.tune_band([c.units[band] for c in channels], band, settings)
        for band in settings.bands
    }


def score(test: dict, tuned: dict, margin_s: float) -> pd.DataFrame:
    """One row of counts per (test channel, detector, band)."""
    rows = []
    for snr, channels in test.items():
        for channel in channels:
            for band, tuned_band in tuned.items():
                detections = benchmark.detect_tuned(tuned_band, channel.units[band])
                for name, det in detections.items():
                    counts = evaluate.score_simulated_band(det, channel.truth, band, margin_s)
                    rows.append(
                        {
                            "snr_db": snr,
                            "seed": channel.seed,
                            "detector": name,
                            "band": band,
                            "minutes": channel.minutes,
                            **counts,
                        }
                    )
    return pd.DataFrame(rows)


def plot(metrics: pd.DataFrame, path: Path) -> None:
    bands = ["ripple", "fast_ripple"]
    fig, axes = plt.subplots(2, len(PANELS), figsize=(15, 6.5), sharex=True, sharey=True)
    for row, band in enumerate(bands):
        for col, (metric, title) in enumerate(PANELS):
            ax = axes[row, col]
            for name, color in DETECTORS.items():
                m = metrics[
                    (metrics.band == band) & (metrics.detector == name) & (metrics.metric == metric)
                ].sort_values("snr_db")
                ax.fill_between(m.snr_db, m.ci_low, m.ci_high, color=color, alpha=0.15, lw=0)
                ax.plot(
                    m.snr_db, m.estimate, color=color, lw=2, marker="o", ms=4, label=LABELS[name]
                )
            ax.set_ylim(-0.03, 1.03)
            if row == 0:
                ax.set_title(title, loc="left")
            if col == 0:
                ax.set_ylabel(
                    "Ripples (80-250 Hz)" if band == "ripple" else "Fast ripples (250-500 Hz)"
                )
            if row == 1:
                ax.set_xlabel("SNR in the event band (dB)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Simulation benchmark: thresholds tuned on training simulations, "
        "shaded = 95 % bootstrap CI over test channels",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    bench = cfg["benchmark"]
    settings = benchmark.settings_from_config(cfg)
    base = SimulationParams.from_config(cfg["simulation"])

    train, test = build_sets(cfg, settings, "uniform", base)
    tuned = tune(train, settings)
    thresholds = pd.DataFrame(
        [
            {
                "band": b,
                "energy_k": t.energy_k,
                "wavelet_z": t.wavelet_z,
                "ml_threshold": t.ml_threshold,
                **{f"train_f1_{k}": v for k, v in t.train_f1.items()},
            }
            for b, t in tuned.items()
        ]
    )
    results = cfg["paths"]["results_dir"]
    thresholds.round(4).to_csv(
        results / "sim_tuned_thresholds.csv", index=False, lineterminator="\n"
    )

    counts = score(test, tuned, bench["spike_margin_s"])
    counts.to_csv(results / "sim_eval_counts.csv", index=False, lineterminator="\n")
    metrics = benchmark.summarise(
        counts,
        ["detector", "band", "snr_db"],
        benchmark.METRICS,
        bench["n_bootstrap"],
        bench["ci_level"],
        bench["bootstrap_seed"],
    )
    metrics.round(4).to_csv(results / "sim_eval_metrics.csv", index=False, lineterminator="\n")
    plot(metrics, cfg["paths"]["figures_dir"] / "sim_benchmark_snr.png")

    print(thresholds.round(3).to_string(index=False))
    table = metrics[
        metrics.metric.isin(["f1", "recall_isolated", "recall_on_spike", "spike_false_alarm_rate"])
    ]
    print(
        table.pivot_table(
            index=["band", "detector"], columns=["metric", "snr_db"], values="estimate"
        )
        .round(2)
        .to_string()
    )
    n_events = (
        counts[counts.detector == "energy"]
        .groupby("band")[["n_isolated", "n_on_spike", "n_spikes"]]
        .sum()
    )
    print("\nTest events:\n", n_events.to_string(), "\n(isolated spikes counted once per band)")
    print(
        f"Units: {counts.seed.nunique()} test channels, "
        f"{np.round(counts.drop_duplicates('seed').minutes.sum(), 1)} min"
    )


if __name__ == "__main__":
    main()
