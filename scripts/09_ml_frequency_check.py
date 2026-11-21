"""Does the ML detector rely on a simulator artefact through the ``freq_hz`` feature?

Hypothesis (step 3): simulated spikes ring near the lower edge of the fast-ripple band
(~250-300 Hz) while simulated fast ripples are uniform over 250-500 Hz, so "high frequency"
would separate them in simulation but not in real data, where fast ripples also sit near the
lower edge.

1. Measure frequencies (zero crossings of the 250-500 Hz signal):
   * of the real reference fast ripples (pre-resection, analysed channels);
   * of the 250-500 Hz ringing at real transients (``results/spike_ringing_real.csv``);
   * of simulated fast ripples and of simulated isolated spikes.
2. Simulate test channels whose fast-ripple frequencies follow the **real** distribution
   ("realistic" test set), and a matching training set.
3. Compare four ML variants on the realistic test set (fast-ripple band):
   A trained on uniform-frequency simulations, all features;
   B same training, without ``freq_hz``;
   C same training, without ``freq_hz`` and ``n_cycles`` (n_cycles = duration x frequency);
   D trained on realistic-frequency simulations, all features.
   If A collapses while B, C and D hold, the ML depended on the artefact.

The real frequency list is computed on all patients; it is only used for this simulation
check, never in the real-data leave-one-patient-out evaluation.

Usage:
    python scripts/09_ml_frequency_check.py [--config configs/default.toml]

Outputs:
    results/frequency_distributions.csv   quantiles of the four frequency distributions
    results/real_fr_frequencies.csv       one row per real reference fast ripple
    results/ml_frequency_check.csv        metrics (95 % CI) per variant, test set and SNR
    figures/ml_frequency_check.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epidetect import benchmark, evaluate, filters, io, plotting, simulate
from epidetect.config import load_config
from epidetect.detectors.base import zero_crossing_frequency
from epidetect.features import FEATURE_NAMES
from epidetect.simulate import SimulationParams

FR_BAND = (250.0, 500.0)
RINGING_HALF_WINDOW_S = 0.01  # spike ringing: zero crossings within +/- 10 ms of the transient
MAX_REAL_TRANSIENTS = 3000  # random subsample of real transients (enough for quantiles)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
VARIANTS = {
    "A: uniform training, all features": ("uniform", tuple(FEATURE_NAMES)),
    "B: uniform training, no freq_hz": (
        "uniform",
        tuple(f for f in FEATURE_NAMES if f != "freq_hz"),
    ),
    "C: uniform training, no freq_hz / n_cycles": (
        "uniform",
        tuple(f for f in FEATURE_NAMES if f not in ("freq_hz", "n_cycles")),
    ),
    "D: realistic training, all features": ("realistic", tuple(FEATURE_NAMES)),
}
VARIANT_COLORS = [plotting.SERIES[2], plotting.SERIES[3], plotting.SERIES[6], plotting.SERIES[0]]


def real_frequencies(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Frequency of every real reference FR, and of the ringing at real transients."""
    excluded = {tuple(x) for x in cfg["data"]["excluded_recordings"]}
    transients = pd.read_csv(
        cfg["paths"]["results_dir"] / "spike_ringing_real.csv", dtype={"patient": str}
    )
    transients = transients.sample(min(MAX_REAL_TRANSIENTS, len(transients)), random_state=0)
    fr_rows, ringing = [], []
    for patient in sorted({p for p, s in io.list_patient_files(cfg["paths"]["raw_dir"]).items()}):
        if (patient, "pre") in excluded:
            continue
        raw = io.read_bids_recording(cfg["paths"]["bids_root"], patient, "pre", cfg["bids"]["task"])
        fs = raw.info["sfreq"]
        markings = io.annotations_to_markings(raw)
        wanted = set(markings.channel) | set(
            transients.loc[transients.patient == patient, "channel"]
        )
        for channel in sorted(wanted - set(raw.info["bads"])):
            y = filters.bandpass(raw.get_data(picks=[channel])[0] / io.MICROVOLT, fs, FR_BAND)
            for m in markings[markings.channel == channel].itertuples():
                seg = y[int(m.onset_s * fs) : int((m.onset_s + m.duration_s) * fs) + 1]
                fr_rows.append(
                    {
                        "patient": patient,
                        "channel": channel,
                        "onset_s": m.onset_s,
                        "freq_hz": zero_crossing_frequency(seg, fs),
                    }
                )
            half = int(RINGING_HALF_WINDOW_S * fs)
            for t in transients.loc[
                (transients.patient == patient) & (transients.channel == channel), "time_s"
            ]:
                c = int(t * fs)
                ringing.append(zero_crossing_frequency(y[c - half : c + half + 1], fs))
    return pd.DataFrame(fr_rows), np.array(ringing)


def simulated_frequencies(params: SimulationParams, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies of simulated fast ripples (truth) and of the ringing at isolated spikes."""
    sim = simulate.simulate_from_seed(params.replace(duration_s=600.0), seed)
    y = filters.bandpass(sim.signal_uv, sim.fs, FR_BAND)
    half = int(RINGING_HALF_WINDOW_S * sim.fs)
    spikes = sim.events[sim.events.kind == "spike"]
    ringing = [
        zero_crossing_frequency(y[int(c * sim.fs) - half : int(c * sim.fs) + half + 1], sim.fs)
        for c in spikes.center_s
    ]
    fr = sim.events.loc[sim.events.kind == "fast_ripple", "freq_hz"].to_numpy()
    return fr, np.array(ringing)


def evaluate_variants(cfg: dict, base: SimulationParams, real_fr_hz: np.ndarray) -> pd.DataFrame:
    """Metrics of each ML variant on the realistic and the uniform test sets (FR band)."""
    bench = cfg["benchmark"]
    cache = benchmark.cache_dir(cfg, "sim")
    realistic = base.replace(events={"fast_ripple_freqs_hz": tuple(np.round(real_fr_hz, 3))})
    sets = {}
    for name, params in [("uniform", base), ("realistic", realistic)]:
        train, test = {}, {}
        for i, snr in enumerate(bench["snr_levels_db"]):
            p = params.replace(duration_s=bench["sim_duration_s"], events={"snr_db": snr})
            n_tr, n_te = bench["n_train_per_snr"], bench["n_test_per_snr"]
            tr = range(
                bench["train_seed_start"] + i * n_tr, bench["train_seed_start"] + (i + 1) * n_tr
            )
            te = range(
                bench["test_seed_start"] + i * n_te, bench["test_seed_start"] + (i + 1) * n_te
            )
            settings = benchmark.settings_from_config(cfg)
            train[snr] = benchmark.simulated_units(
                p, tr, settings, cache, f"{name}-train-snr{snr:g}"
            )
            test[snr] = benchmark.simulated_units(p, te, settings, cache, f"{name}-test-snr{snr:g}")
        sets[name] = (train, test)

    rows = []
    for variant, (training, features) in VARIANTS.items():
        settings = benchmark.settings_from_config(cfg, feature_names=features)
        train_units = [
            c.units["fast_ripple"] for units in sets[training][0].values() for c in units
        ]
        tuned = benchmark.tune_band(train_units, "fast_ripple", settings)
        for test_name, (_, test) in sets.items():
            for snr, channels in test.items():
                for channel in channels:
                    det = benchmark.detect_tuned(tuned, channel.units["fast_ripple"])["ml"]
                    counts = evaluate.score_simulated_band(
                        det, channel.truth, "fast_ripple", bench["spike_margin_s"]
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "test_set": test_name,
                            "snr_db": snr,
                            "seed": channel.seed,
                            "minutes": channel.minutes,
                            "ml_threshold": tuned.ml_threshold,
                            **counts,
                        }
                    )
    counts = pd.DataFrame(rows)
    metrics = {
        k: benchmark.METRICS[k]
        for k in ("f1", "precision", "recall_isolated", "recall_on_spike", "spike_false_alarm_rate")
    }
    return benchmark.summarise(
        counts,
        ["variant", "test_set", "snr_db"],
        metrics,
        bench["n_bootstrap"],
        bench["ci_level"],
        bench["bootstrap_seed"],
    )


def plot(distributions: dict[str, np.ndarray], metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    bins = np.linspace(200, 550, 36)
    styles = [
        (plotting.INK, "-"),
        (plotting.NEUTRAL, "-"),
        (plotting.SERIES[0], "--"),
        (plotting.ACCENT, "--"),
    ]
    for (name, values), (color, ls) in zip(distributions.items(), styles, strict=True):
        values = values[np.isfinite(values)]
        hist, edges = np.histogram(values, bins=bins, density=True)
        axes[0].step(
            edges[:-1],
            hist,
            where="post",
            color=color,
            ls=ls,
            lw=1.5,
            label=f"{name} (median {np.median(values):.0f} Hz)",
        )
    axes[0].set_xlabel("Frequency in 250-500 Hz (zero crossings)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Fast ripples and spike ringing: real vs simulated", loc="left")
    axes[0].legend(fontsize=7)
    for ax, test_set, title in [
        (axes[1], "realistic", "Realistic-frequency test set"),
        (axes[2], "uniform", "Uniform-frequency test set"),
    ]:
        for (variant, _), color in zip(VARIANTS.items(), VARIANT_COLORS, strict=True):
            m = metrics[
                (metrics.variant == variant)
                & (metrics.test_set == test_set)
                & (metrics.metric == "f1")
            ].sort_values("snr_db")
            ax.fill_between(m.snr_db, m.ci_low, m.ci_high, color=color, alpha=0.12, lw=0)
            ax.plot(m.snr_db, m.estimate, color=color, marker="o", ms=4, lw=1.8, label=variant)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("F1, fast ripples")
        ax.set_title(title, loc="left")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    plotting.apply_style()
    results = cfg["paths"]["results_dir"]
    base = SimulationParams.from_config(cfg["simulation"])

    real_fr, real_ringing = real_frequencies(cfg)
    real_fr.round(3).to_csv(results / "real_fr_frequencies.csv", index=False, lineterminator="\n")
    sim_fr, sim_ringing = simulated_frequencies(base, cfg["simulation"]["seed"])
    distributions = {
        "real fast ripples": real_fr["freq_hz"].to_numpy(),
        "real transients (ringing)": real_ringing,
        "simulated fast ripples (uniform)": sim_fr,
        "simulated spikes (ringing)": sim_ringing,
    }
    table = pd.DataFrame(
        {name: np.nanquantile(v, QUANTILES) for name, v in distributions.items()},
        index=[f"q{int(q * 100)}" for q in QUANTILES],
    ).T
    table["n"] = [int(np.isfinite(v).sum()) for v in distributions.values()]
    table.round(1).to_csv(results / "frequency_distributions.csv", lineterminator="\n")
    print(table.round(1).to_string())

    metrics = evaluate_variants(cfg, base, real_fr["freq_hz"].dropna().to_numpy())
    metrics.round(4).to_csv(results / "ml_frequency_check.csv", index=False, lineterminator="\n")
    plot(distributions, metrics, cfg["paths"]["figures_dir"] / "ml_frequency_check.png")
    print(
        metrics[metrics.metric == "f1"]
        .pivot_table(index=["test_set", "variant"], columns="snr_db", values="estimate")
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
