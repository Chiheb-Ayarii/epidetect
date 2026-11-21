# EpiDetect

Reproducible benchmark of high-frequency oscillation (HFO) detectors on realistic simulations
and on public intraoperative ECoG recordings.

> Work in progress. The full README (results, figures, limitations) is written at the end of the
> project. Data notes: [docs/DATA_NOTES.md](docs/DATA_NOTES.md).

## Data

The data are **not** stored in this repository. Download the 22 `Data_Patient_XX.h5` files from
<https://gin.g-node.org/USZ_NCH/Intraoperative_ECoG_HFO> (folder `data/`) into `data/raw/`.

## License

The **code** in this repository is released under the [MIT License](LICENSE)
(Copyright (c) 2026 Chiheb Ayari). The MIT license covers the code only: the **data** are not
part of this repository and keep their own license, **CC BY-SA 4.0**, set by their authors
(Boran et al., 2019, University Hospital Zurich). Any derived data (e.g. the BIDS conversion
produced by `scripts/01_convert_to_bids.py`) must also be shared under CC BY-SA 4.0 with
attribution.

## Quick start (Windows PowerShell)

```powershell
.\run.ps1 setup       # create .venv and install pinned dependencies
.\run.ps1 run         # run every numbered script in scripts/ in order (see below)
.\run.ps1 notebooks   # re-execute the notebooks in notebooks/
.\run.ps1 test        # unit tests (no real data needed)
.\run.ps1 lint
```

| Script | What it produces |
|---|---|
| `scripts/00_inventory_raw.py` | `docs/data_inventory.md`, a read-only inventory of the raw NIX files |
| `scripts/01_convert_to_bids.py` | `data/bids/` (BIDS iEEG, git-ignored), `results/bids_conversion.csv` |
| `scripts/02_quality_control.py` | `results/qc_*.csv`, `figures/qc_*.png`, bad channels marked in BIDS |
| `scripts/03_calibrate_simulator.py` | background spectrum and spike sharpness fitted on the real data: `results/background_fit*.csv`, `results/spike_*.csv`, `figures/sim_*_calibration.png` |
| `scripts/04_simulation_examples.py` | `figures/sim_examples.png`, `figures/sim_snr_levels.png`, `results/sim_example_truth.csv` |
| `scripts/05_spike_filter_demo.py` | why a filtered spike looks like an HFO: `figures/false_hfo_demo.png`, `results/false_hfo_demo.csv` |
| `scripts/06_notch_decision.py` | effect of notch-filtering mains harmonics on real detections: `results/notch_decision.csv` (~6 min) |
| `scripts/07_detector_examples.py` | the three detectors on four typical simulated situations: `figures/detector_examples.png` |
| `scripts/08_evaluate_simulations.py` | simulation benchmark per class, SNR and event sub-group: `results/sim_eval_*.csv`, `figures/sim_benchmark_snr.png` (~5 min) |
| `scripts/09_ml_frequency_check.py` | does the ML detector rely on a simulator frequency artefact? `results/ml_frequency_check.csv`, `figures/ml_frequency_check.png` (~5 min) |
| `scripts/10_evaluate_real.py` | leave-one-patient-out on the real fast-ripple markings, two tuning strategies: `results/real_lopo_*.csv`, `figures/real_lopo.png` (~40 min) |
| `scripts/11_real_fr_context.py` | share of real fast ripples riding on a large transient: `results/real_fr_context.csv` |
| `scripts/12_evaluate_hybrid.py` | strategy C, tuned on a hybrid simulation (synthetic FRs in real background of the training patients), compared with A and B: `results/strategy_*.csv`, `figures/strategy_comparison.png` (~40 min) |
| `scripts/13_clinical.py` | patient-level clinical analysis (post-resection FR rate vs outcome): `results/clinical_*.csv`, `figures/clinical_roc.png` (~30 min) |

Channel-level clinical analysis (optional): if a file `data/resection_labels.csv` with columns
`patient,channel,resected` (patient as two digits, e.g. `01`; channel as in BIDS, e.g. `01-02`;
`resected` as `True`/`False`) is provided, `scripts/13_clinical.py` also computes, for each
patient, the AUC of pre-resection channel FR rates against the resection labels. No such labels
exist for the public dataset.

Intermediate analyses are cached in `data/derived/` (git-ignored); the cache folder name
contains a hash of the configuration, so changing a parameter never reuses stale results.

The BIDS output passes the official validator (requires Node.js):

```powershell
npx bids-validator@1.15.0 data/bids
# -> "This dataset appears to be BIDS compatible": 0 errors, 0 warnings
#    (488 files, 902 MB, 22 subjects, 2 sessions, iEEG, task "intraop")
```

## Limitations

This section is completed at every step; the final README discusses each point with the results.

**Data**
- Only **fast ripples** are marked in the real data, by an automatic detector followed by visual
  validation. This reference is detector-biased, and ripples can only be evaluated on
  simulations.
- No channel-level resection labels exist, so the clinical analysis is done at the patient level.
- The pre-resection recordings of patients 12, 15 and 17 are byte-identical in the source files.
  They are excluded.
- **P22 pre-resection, channel 3-4** contains duplicated samples: every sample appears twice,
  so it was effectively sampled at 1000 Hz and up-sampled to 2000 Hz. Its effective Nyquist
  frequency is 500 Hz, so it cannot carry fast ripples. It is flagged by the quality control
  and excluded from fast-ripple analyses.
- Small cohort: 22 patients, of whom 8 were not seizure-free.

**Simulator**
- The background spectrum (1/f^β with a knee, plus a white amplifier floor) and the spike-apex
  sharpness are calibrated on the real data. The spike shape itself (amplitudes, time
  constants, slow wave) is a modelling choice.
- The real anti-aliasing roll-off above 500 Hz and mains interference are not simulated.
- One channel per simulation: no spatial correlation between channels.
- The spike-apex calibration matches the **amplitude** of the 250–500 Hz ringing of real
  transients, but not its **frequency**. Simulated spikes ring at a median of 268 Hz, against
  341 Hz for real transients and 303 Hz for real fast ripples
  (`results/frequency_distributions.csv`). The rounded-apex model makes the spike spectrum fall
  too steeply inside the band.
- The SNR is defined against the background power of the **whole** band. With a 1/f^3.5
  background, most of the 80–250 Hz background power lies near 80 Hz, so a 200 Hz ripple "at
  0 dB" is locally well above the background. SNR values are therefore not comparable between
  the ripple and fast-ripple bands, and they favour detectors that work locally in frequency
  (the whitened wavelet map).
- For HFOs riding on a spike, a detection may be triggered by the spike rather than by the
  oscillation: the energy detector "finds" 100 % of them in the ripple band even at 0 dB.
  Recall on this sub-group must be read together with the false-alarm rate on isolated
  spikes.
- The simulator puts 30 % of HFOs on a spike. In the real data, 62 % of the reference fast
  ripples (pooled; median 30 % per patient) lie within 50 ms of a large transient
  (`results/real_fr_context.csv`).
- Thresholds tuned on simulations do not transfer to the real data. They are far too
  permissive (energy k = 3.5 versus 8, wavelet z = 6 versus 12–15 when tuned on real training
  patients), giving 0.7–1.6 false detections per minute in patients without reference fast
  ripples (`results/real_lopo_summary.csv`, strategy B). Real recordings contain many more
  fast-ripple-band transients than the simulator.

**Real-data evaluation**
- Only 16 patients have reference fast ripples, and 7 of them have 1 to 13. Every detector
  gets F1 = 0 on those 7. Macro-averaged F1 values have wide bootstrap intervals; for
  example, ML 0.25 [0.12, 0.39].
- Micro-averaged (pooled) metrics are dominated by patient 06 (1,413 of 2,378 FRs) and are
  much higher than macro averages; for example, ML micro F1 0.61 versus macro 0.25.
- The calibration of the simulator used every patient. Recalibrating it without the test
  patient changes its parameters by about 1–8 % (`results/real_fold_calibration.csv`), so this
  leakage is negligible. Strategy B uses the recalibrated simulator in every fold anyway.
- Hybrid simulation (strategy C: synthetic fast ripples inserted into real background of the
  training patients, with rate, share on spikes, frequency, duration and SNR calibrated on
  their markings) brings the tuned thresholds close to those tuned on real data. Median energy
  k is 7, versus 8 for A and 3.5 for B; median wavelet z is 12, versus 12 for A and 6 for B. For
  the energy and wavelet detectors, it matches strategy A in macro F1 (paired difference
  +0.003). The gain over the fully synthetic simulator (+0.02 to +0.03) has a confidence
  interval that includes 0. The ML detector trained on hybrid data fails (macro recall 0.09).
  A plausible but untested explanation: realistic noise is enough to set a threshold, but a
  classifier also needs realistic positive examples. The inserted Hann-windowed sinusoids are
  cleaner than real FRs, and unmarked real FRs in the "background" are learned as negatives
  (`results/strategy_comparison.csv`).
**Clinical analysis** (exploratory: 22 patients, 8 not seizure-free)
- With the reference markings, the post-resection FR rate predicts a poor outcome: AUC 0.83
  [0.62, 1.00], permutation p = 0.004. This reproduces the original publication's finding on
  the public data. The files do not say whether the markings were made blind to the outcome.
- With our three detectors, post-resection rates do not predict the outcome (AUC 0.40–0.61,
  every CI includes 0.5; `results/clinical_auc.csv`). Their false positives in seizure-free
  patients drown the signal, so a detector agreeing with the reference at macro F1 ≈ 0.2 is
  not clinically usable as it is.
- Confidence intervals come from a stratified bootstrap over 22 patients and are approximate.
  Their upper bound reaches 1.
- Channel-level analysis (FR rate vs resected channels) is implemented but not run: no
  channel-level resection labels exist.

**Data dominance**
- A single patient dominates the dataset statistics: without patient 06, the calibrated FR
  rate drops from 2.5 to 1.1 per channel-minute and the share on spikes from 62 % to 39 %
  (`results/hybrid_fold_calibration.csv`).

**Detectors** (behaviour observed on simulations in step 3, quantified in step 4)
- Energy detector: sharp spikes produce detections, especially in the ripple band, where a
  spike's spectrum is strong. No threshold removes them without losing HFOs.
- Wavelet detector: its spike-rejection test also rejects HFOs riding on a spike, which look
  like a spike column with a bump in the time–frequency map.
- ML detector: its most important fast-ripple feature is the frequency. This may partly
  reflect a simulator property (spikes ring near the lower edge of the band) and must be
  checked on real data.
- No notch filter is applied: mains harmonics add only about 1–3 % of detections on the
  affected real channels (`results/notch_decision.csv`).
