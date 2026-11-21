**Simulations, SNR 10 dB** (`results/sim_eval_metrics.csv`; 95 % bootstrap CI over 10 test channels of 120 s)

| Band | Detector | F1 | Precision | Recall, isolated HFOs | Recall, HFOs on a spike | Isolated spikes detected |
|---|---|---|---|---|---|---|
| Fast ripples | Energy (RMS) | 0.68 [0.64, 0.72] | 0.61 | 0.72 | 0.88 | 25 % |
| Fast ripples | Wavelet (Morlet) | 0.81 [0.76, 0.86] | 0.92 | 0.98 | 0.09 | 0 % |
| Fast ripples | ML (random forest) | 0.87 [0.82, 0.92] | 0.89 | 0.99 | 0.52 | 0 % |
| Ripples | Energy (RMS) | 0.50 [0.44, 0.56] | 0.37 | 0.71 | 1.00 | 100 % |
| Ripples | Wavelet (Morlet) | 0.78 [0.71, 0.81] | 0.96 | 0.97 | 0.00 | 1 % |
| Ripples | ML (random forest) | 0.75 [0.70, 0.80] | 0.81 | 0.95 | 0.19 | 13 % |

**Real data, fast ripples, leave-one-patient-out** (`results/strategy_comparison.csv`; 16 patients with reference FRs, 95 % bootstrap CI over patients)

| Strategy | Detector | Macro F1 | Micro F1 | FP / min (3 patients without FR) | Median threshold |
|---|---|---|---|---|---|
| A: tuned on real training patients | Energy (RMS) | 0.19 [0.09, 0.32] | 0.52 | 0.30 | k = 8 |
| A: tuned on real training patients | Wavelet (Morlet) | 0.11 [0.06, 0.19] | 0.30 | 0.06 | z = 12 |
| A: tuned on real training patients | ML (random forest) | 0.25 [0.12, 0.39] | 0.61 | 0.00 | p = 0.4 |
| B: tuned on synthetic simulator | Energy (RMS) | 0.18 [0.11, 0.26] | 0.27 | 0.69 | k = 3.5 |
| B: tuned on synthetic simulator | Wavelet (Morlet) | 0.09 [0.05, 0.14] | 0.11 | 1.52 | z = 6 |
| B: tuned on synthetic simulator | ML (random forest) | 0.14 [0.08, 0.21] | 0.22 | 1.55 | p = 0.4 |
| C: tuned on hybrid simulation | Energy (RMS) | 0.20 [0.09, 0.32] | 0.49 | 0.33 | k = 7 |
| C: tuned on hybrid simulation | Wavelet (Morlet) | 0.12 [0.06, 0.19] | 0.30 | 0.06 | z = 12 |
| C: tuned on hybrid simulation | ML (random forest) | 0.11 [0.03, 0.21] | 0.10 | 0.00 | p = 0.55 |

**Clinical, patient level** (`results/clinical_auc.csv`; post-resection FR rate vs not seizure-free; 8 vs 14 patients; stratified bootstrap CI)

| Score | AUC [95 % CI] | Permutation p |
|---|---|---|
| Reference markings | 0.83 [0.62, 1.00] | 0.004 |
| Energy (RMS) detector | 0.45 [0.21, 0.70] | 0.701 |
| Wavelet (Morlet) detector | 0.61 [0.36, 0.84] | 0.411 |
| ML (random forest) detector | 0.40 [0.18, 0.64] | 0.455 |

Rule "at least one reference FR after resection": sensitivity 6/8, specificity 10/14 (`results/clinical_rule.csv`).
