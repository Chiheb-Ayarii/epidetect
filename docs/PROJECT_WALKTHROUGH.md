# EpiDetect – Project walkthrough

A step-by-step account of the whole project, from an empty folder to the clinical analysis.
Every number in this document comes from a file in `results/`, produced by the scripts of the
repository (`.\run.ps1 run`).

---

## The project in 2 minutes

**Question.** High-frequency oscillations (HFOs) in intracranial EEG, especially fast ripples
(FR, 250–500 Hz), are candidate biomarkers of epileptogenic tissue. Many automatic detectors
exist. How good are they really, and where do they fail?

**Data.** A public dataset from the University Hospital Zurich: intraoperative ECoG of 22
patients, recorded before and after the resection, sampled at 2000 Hz. It includes fast-ripple
markings (an automatic detector followed by visual validation) and the surgical outcome
(14 seizure-free, 8 not). I converted it to BIDS (0 errors in the official validator) and
quality-controlled every channel.

The quality control found three anomalies:
- three byte-identical recordings attributed to different patients;
- one channel with duplicated samples, i.e. effectively sampled at 1000 Hz;
- 18 noisy channels out of 429.

**Simulator.** I built a simulator with exact ground truth: a 1/f background with an
amplifier floor, ripples, fast ripples and spikes. I calibrated it on the real data: 1/f
exponent 3.48, crossover with the floor at 260 Hz, and the spike sharpness matched on 8,630
real transients. A deterministic demonstration shows the central trap of the field: after a
250–500 Hz band-pass, a sharp −500 µV spike gives a 26 µV "oscillation", six times a genuine
5 µV fast ripple.

**Three detectors.**
1. An RMS-energy detector (Staba-style).
2. A whitened Morlet time–frequency detector that rejects spikes with a "blob or column" test
   (principle of Roehri 2016 / Delphos, own code).
3. A random forest on 11 features of permissive candidates.

**Evaluation.**
- *Simulations* (per class, per SNR, per sub-group, with bootstrap CIs). At 10 dB the ML
  detector has the best fast-ripple F1 (0.87). The wavelet detector is the most precise but
  misses HFOs riding on spikes (recall 0.09). The energy detector fires on every isolated spike
  in the ripple band.
- *Real data* (leave-one-patient-out, nothing tuned on the test patient). The best detector,
  ML, reaches macro F1 0.25 [0.12, 0.39] against the reference, versus 0.61 when events are
  pooled (dominated by one patient). Every detector fails on the 7 patients with few FRs.
  - Thresholds tuned on the synthetic simulator do not transfer to real data.
  - A **hybrid simulation** (synthetic FRs in real background of the training patients) fixes
    the thresholds of the energy and wavelet detectors (median k 7 vs 8 on real data, z 12 vs
    12), but not the ML detector.

**Clinical.** With the reference markings, the post-resection FR rate predicts a poor outcome:
AUC 0.83 [0.62, 1.00], permutation p = 0.004. This reproduces the original publication. With
our detectors it does not (AUC 0.40–0.61), because their false positives drown the signal.

**Take-away.** The hard part of HFO detection is not the detector but the evaluation: what the
reference is, which unit is independent, how to tune without leakage, and whether simulation
results transfer to real data.

---

## How to reproduce

```powershell
.\run.ps1 setup       # .venv + pinned dependencies
.\run.ps1 run         # scripts/00 ... 13, in order (about 2 h without cache)
.\run.ps1 notebooks   # re-executes notebooks/01 ... 05
.\run.ps1 test        # 87 tests, no real data needed
```

The raw data go in `data/raw/` (see the README). Everything derived from them (`data/bids`,
`data/derived`) is git-ignored; small result tables and figures are committed.

---

## Step 0 – Environment and data inspection

### What we did and why
We created the Python 3.11 environment, the project structure and the task runner (`run.ps1`,
since `make` is not available on Windows). Then we **inspected the raw files before writing any
loader**, and wrote what we found in `docs/DATA_NOTES.md`. The point is to understand the data
format, its units, its labels and their caveats before trusting any analysis.

### Key concept: the NIX format and its pitfalls
The files are HDF5 written with the NIX data model: blocks (pre/post resection), data arrays
with explicit dimensions, tags for annotated intervals, and a metadata tree. Two pitfalls were
found:
1. The fast-ripple tags store a **1-based** channel index (MATLAB style), whereas NIX indices
   are 0-based. This was checked on all 2,479 tags.
2. Some metadata strings are stored one character per record.

### Key code (`scripts/00_inventory_raw.py`, simplified)
```python
with h5py.File(path, "r") as f:
    block = f["data"][block_name]
    array = block["data_arrays/ECoG_Bipolar_Channels"]
    n_channels, n_samples = array["data"].shape            # signals in uV
    dt = array["dimensions/2"].attrs["sampling_interval"]  # 0.0005 s -> 2000 Hz
    tags = block["groups/FR markings/tags"]                # one tag per fast ripple
```

### Choices and alternatives
We read the files with plain `h5py` rather than the `nixio` library: it adds no dependency and
shows the layout explicitly. The inventory is a script, not a notebook, so the numbers quoted
in `DATA_NOTES.md` can be regenerated.

### Results
- 22 patients: 14 seizure-free (ILAE 1), 8 not.
- 2,378 pre-resection and 101 post-resection FR markings.
- Patient 06 holds about 59 % of the pre-resection FRs.
- The markings come from an automatic detector plus visual validation, so the reference is
  detector-biased.
- Only fast ripples are marked, and there are no channel-level resection labels.

### Common mistakes
- Trusting the NIX index convention without checking it against channel names.
- Pooling all events and forgetting that one patient dominates.

### Exercise
Open one tag in `h5py` and check by hand that `labels[index − 1]` equals the channel label in
the tag name.

---

## Step 1 – Loading, BIDS conversion, quality control

### What we did and why
`epidetect.io` reads the NIX files and converts them to BIDS with `mne-bids`. Every recording
is read back and compared with the source. `epidetect.qc` then computes, for every channel:
- a median Welch spectrum;
- the 1/f exponent;
- the height of every mains harmonic;
- robust outlier flags;
- a duplicate check across recordings.

### Key concepts
- **Welch PSD**: the periodogram of one long segment is not a consistent estimator (its
  variance does not decrease). Averaging 1 s Hann windows trades frequency resolution for
  variance. We use the **median** over windows, which ignores the few windows containing fast
  ripples. Note that its bias correction (÷ ln 2) is exact for Gaussian noise but inflates pure
  tones by 1.44, so mains harmonics are measured on the mean PSD.
- **Robust z-score**: (x − median) / (1.4826 × MAD). One aberrant channel cannot hide itself by
  inflating the standard deviation.
- **BIDS**: a folder and file naming standard (`sub-01/ses-preresection/ieeg/...`) with
  `channels.tsv`, `events.tsv`, `participants.tsv`, so that any tool can read the data.

### Key code (`src/epidetect/io.py`)
```python
index_1based, onset_s = (float(v) for v in tag["position"][()])
channel = ch_names[int(index_1based) - 1]          # 1-based, verified on every tag
label_in_name = str(tag.attrs["name"]).split("_")[-1]
if label_in_name != channel:                        # guards the convention forever
    raise ValueError(...)
```
`mne-bids` dropped the channel of channel-specific annotations, so `events.tsv` is rewritten
with an extra `channel` column, documented in `events.json`.

### Choices and alternatives
- **BrainVision float32 storage** rather than EDF, which would have needed 16-bit rescaling.
- **Bad channels are marked, never deleted.**
- **Relative QC criteria only with at least 8 channels**: with 4–6 strip channels the MAD is
  tiny and z-scores reached −43.

### Results
- The official BIDS validator reported 0 errors and 0 warnings on 488 files. The round-trip
  error was at most 3.9·10⁻⁴ µV.
- The pre-resection recordings of patients 12, 15 and 17 are **byte-identical**. They are
  excluded, and the exclusion is justified automatically by a hash check.
- 18 of 429 channels are flagged. One of them, P22 pre 3-4, has **duplicated samples**
  (effectively 1000 Hz), so it cannot carry fast ripples. It was found in step 2 and added to
  the QC.
- The 2–40 Hz exponent β has a median of 2.55. Above about 150–250 Hz the spectrum flattens:
  this is the amplifier noise floor.
- The mains interference is small. P14 shows only even harmonics, which suggests a 100 Hz
  source such as rectified mains. P20 after resection shows only odd harmonics, the signature
  of half-wave symmetry.

### Common mistakes
- Treating the MNE data as microvolts (MNE stores volts).
- Flagging FR-rich channels as "noisy" with a mean PSD.
- Deleting bad channels instead of marking them.

### Exercise
Simulate a 50 Hz sine plus noise and compare its 50 Hz peak on the mean and on the median Welch
PSD. Explain the factor 1.44.

---

## Step 2 – Digital filters and the simulator

### What we did and why
A zero-phase band-pass (`epidetect.filters`) and a simulator with exact ground truth
(`epidetect.simulate`), **calibrated on the real data**
(`scripts/03_calibrate_simulator.py`). Then a deterministic demonstration of why a filtered
spike looks like an HFO (`scripts/05_spike_filter_demo.py`).

### Key concepts
- **From analogue to digital filters.** A digital filter is a difference equation.
  - FIR filters have finite impulse responses and can be linear-phase.
  - IIR filters use feedback, like RC/RLC circuits, and are steep with few coefficients.
  - Second-order sections keep high-order IIR filters numerically stable.
  - `filtfilt` (forward then backward) gives |H|²: zero phase, and −6 dB at the cut-offs.
- **Why a filtered spike looks like an HFO.** The impulse response of a band-pass filter is a
  short oscillation. A spike with an angular apex has a 1/f² spectrum, so energy at 250–500 Hz,
  and the filter turns it into a copy of its impulse response. A smooth spike has a negligible
  spectrum there. Narrower bands and steeper filters give longer false oscillations (Gibbs
  tails, time–frequency uncertainty).
- **1/f noise generation.** Multiply the FFT of white noise by √S(f), then take the inverse FFT.

### Key code (`src/epidetect/simulate.py`, simplified)
```python
spectrum = np.fft.rfft(rng.standard_normal(n_samples))
freqs = np.fft.rfftfreq(n_samples, 1 / fs)
spectrum *= np.sqrt(background_psd_shape(freqs, bg))   # 1/f^beta (knee at 20 Hz) + floor
x = np.fft.irfft(spectrum, n=n_samples)
background = bg.std_uv * x / x.std()
```
```python
# Spike: exponential cusp, apex rounded by a Gaussian of width apex_sigma (calibrated)
cusp = np.where(t < 0, np.exp(t / tau_rise), np.exp(-t / tau_decay))
cusp = np.convolve(cusp, gaussian_kernel, mode="same")
```

### Choices and alternatives
- **SNR defined in the event's band**, as in Roehri et al. (2017): mean power of the event
  divided by the background power in the band. Known flaw: with a steep 1/f background it is
  not comparable between bands.
- **Event times from a renewal process** (a minimum gap plus an exponential interval): events
  never overlap, so the matching is unambiguous.

### Results
- **Background**: median exponent 3.48 above 20 Hz (2.5 below) and crossover 260 Hz, over
  channels whose floor is reached below 500 Hz (61 % of 411). The simulated spectrum follows the
  real median from about 8 to 500 Hz.
- **Spikes**: a perfect cusp rang about 20 times too strongly (median ratio 88 versus 4.3 on real
  transients). An apex rounding of 1.1–1.6 ms matches the real quantiles (median 4.8 versus 4.3).
- **False HFO demonstration** (250–500 Hz):

  | Input | Filtered peak | Cycles above 50 % of the envelope |
  |---|---|---|
  | Angular spike, −500 µV | 26.1 µV | 1 |
  | Genuine 5 µV fast ripple | 4.4 µV | 3.5 |
  | Smooth Gaussian spike, −500 µV | about 0 | – |
  | Realistically rounded spike | 1.1 µV | 2.5 |
  | Angular spike, 280–320 Hz band | – | 8 |

### Common mistakes
- Assuming the amplitude of a spike, rather than its sharpness, creates false HFOs.
- Forgetting that `filtfilt` gives −6 dB at the cut-offs.
- Calibrating on transients that contain real FRs (circular reasoning).

### Exercise
Filter a unit impulse in 250–500 Hz and in 280–320 Hz, and count the cycles above 10 % of the
envelope. Relate the result to 1 / bandwidth.

---

## Step 3 – The three detectors

### What we did and why
Three families of detectors, all returning the same table (`onset_s, offset_s, peak_s, band,
freq_hz, score`), plus the one-to-one matching used to label and evaluate them.

### Key concepts
- **Hilbert envelope**: the modulus of the analytic signal. It is only meaningful after a
  band-pass filter.
- **Morlet wavelet**: a Gaussian band-pass with σ_f = f / 6 cycles, i.e. constant Q. It is long
  for ripples and short for fast ripples.
- **Whitening**: a robust z-score per frequency, which makes the 1/f map flat.
- **"Blob or column"**: an HFO is isolated in frequency; a spike is broadband.

### Key code (`src/epidetect/detectors/wavelet.py`)
```python
# Morlet transform in the frequency domain: one Gaussian gain per analysis frequency
gain = np.where(f_axis > 0, 2.0 * np.exp(-0.5 * ((f_axis - f0) / (f0 / n_cycles)) ** 2), 0.0)
out[k] = np.fft.ifft(spectrum * gain)[:n]      # |W| = A for A cos(2 pi f0 t)
```
```python
# Valley test: z just below the peak (0.5-0.65 f_peak), relative to the peak
region = (freqs >= 0.5 * f_peak) & (freqs <= 0.65 * f_peak)
ratio = z_column[region].min() / z_column[f_index]   # ~0 for an HFO, ~0.75 for a spike
```

### Choices and alternatives
- **Energy detector**: median + k × MAD instead of mean + k × SD, so HFO-rich channels do not
  raise their own threshold.
- **Wavelet detector**:
  - the frequency is estimated on the **non-whitened** excess amplitude, because whitening
    biased it upwards (+6 % at 80–150 Hz);
  - my first valley test, searched down to 40 Hz, was wrong: after whitening a spike's z-score
    *decreases* towards low frequencies.
- **ML detector**: a random forest with balanced class weights. Feature extraction is separate
  from training, so leave-one-patient-out does not recompute features.
- **No notch filter**: measured on 72 real channels, harmonics add only about 1–3 % of
  detections.

### Results (simulation, SNR 10 dB)

| | Isolated HFOs | HFOs on a spike | Isolated spikes |
|---|---|---|---|
| Energy | detected | detected (thanks to the spike) | false alarm |
| Wavelet | detected | rejected | rejected |
| ML | detected | partly detected | rejected |

### Common mistakes
- Computing an envelope without band-passing first.
- Estimating frequencies on a whitened map.
- Duplicated feature columns in a concatenated table (here `freq_hz`, caught by a shape error).

### Exercise
Plot the valley ratio of 50 simulated spikes and 50 isolated HFOs, then choose a threshold.
Then add 50 HFOs riding on spikes and explain what happens.

---

## Step 4 – Evaluation on simulations and on real data

### What we did and why
- **Simulations**: 5 SNR levels × (4 training + 10 test channels of 120 s), disjoint seeds.
  Thresholds are tuned on the training channels, and test metrics come with bootstrap CIs over
  test channels.
- **Real data**: leave-one-patient-out on the pre-resection FR markings, with two strategies.
  In A, everything is tuned on the other patients. In B, everything is tuned on simulations
  from a simulator **recalibrated without the test patient**.
- A check of whether the ML detector depends on a simulator artefact (frequency).

### Key concepts
- Precision, recall, F1; **no accuracy**, since FRs cover 0.079 % of the recorded time.
- Micro versus macro averaging.
- Nested tuning: an inner GroupKFold by patient chooses the ML probability threshold.
- Percentile bootstrap over **independent units** (patients).

### Key code (`src/epidetect/benchmark.py`, simplified)
```python
# ML threshold chosen on out-of-fold predictions; references never captured by a candidate
# are false negatives whatever the threshold.
probabilities = out_of_fold_probabilities(table, groups=patients, n_splits=5)
for t in grid:
    tp = ((p >= t) & (label == 1)).sum()
    fp = ((p >= t) & (label == 0)).sum()
    f1 = 2 * tp / (2 * tp + fp + (n_reference - tp))
```

### Results
**Simulation, fast ripples at 10 dB (95 % CI over 10 test channels)**

| Detector | F1 | Recall, isolated | Recall, on a spike | Spike false alarms |
|---|---|---|---|---|
| Energy | 0.68 [0.64, 0.72] | 0.72 | 0.88 | 25 % |
| Wavelet | 0.81 [0.76, 0.86] | 0.98 | 0.09 | 0 % |
| ML | 0.87 [0.82, 0.92] | 0.99 | 0.52 | 0 % |

In the ripple band, the energy detector fires on 100 % of isolated spikes.

**Real data, strategy A**

| Detector | Macro F1 [95 % CI] | Micro F1 | FP per minute, patients without FR |
|---|---|---|---|
| ML | 0.250 [0.124, 0.386] | 0.608 | 0.00 |
| Energy | 0.194 [0.088, 0.318] | 0.521 | 0.30 |
| Wavelet | 0.115 [0.055, 0.185] | 0.303 | 0.06 |

- **Strategy B** (synthetic simulator) has thresholds far too low (k 3.5, z 6), hence many false
  positives (0.7–1.6 per minute) and macro F1 0.09–0.18.
- **Leakage through the calibration** is negligible: without the test patient, the parameters
  change by 1–8 %.
- **Frequency artefact**: simulated spikes ring at 268 Hz versus 341 Hz for real transients,
  yet the ML detector does not depend on `freq_hz` (same F1 without it).
- **Context of real FRs**: 62 % lie within 50 ms of a large transient, which explains the poor
  real-data recall of the wavelet detector.

### Common mistakes
- Reporting a pooled F1.
- Tuning the ML threshold on in-sample predictions.
- Bootstrapping events.
- Comparing recall on HFOs riding on spikes without the spike false-alarm rate.

### Exercise
From `results/real_lopo_per_patient.csv`, recompute the macro F1 of the ML detector, then drop
patient 06 and recompute the micro F1. Explain the difference.

---

## Step 4b – Hybrid simulation

### What we did and why
Strategy C: in each fold, synthetic fast ripples are inserted into real background windows
(30 s, without marked FR) of the **training patients only**. Their properties are calibrated on
the training patients' markings:
- the rate;
- the share riding on a spike, where a "spike" FR is centred on a *real* transient;
- the frequency/duration pairs;
- the in-band SNR = 10·log10(P_marking / P_background − 1).

### Key code (`src/epidetect/hybrid.py`, simplified)
```python
background_power = robust_std(bandpass(window, fs, (250, 500))) ** 2
amplitude = np.sqrt(10 ** (snr / 10) * background_power / np.mean(unit ** 2))
signal[start:start + len(unit)] += amplitude * unit     # unit = Hann-windowed sinusoid
```

### Results (`results/strategy_comparison.csv`)
- **Median thresholds**: k = 7 (A 8, B 3.5) and z = 12 (A 12, B 6).
- **Macro F1, C versus A versus B**:
  - energy 0.196 / 0.194 / 0.177;
  - wavelet 0.118 / 0.115 / 0.088;
  - ML 0.109 / 0.250 / 0.136.
- **Paired difference C − B**: +0.02 [−0.04, 0.08] for energy and +0.03 [−0.04, 0.10] for
  wavelet. The hybrid simulation fixes the **thresholds** but not the **classifier**: realistic
  noise is enough for a threshold, while a classifier also needs realistic positives (domain
  shift, and label noise from unmarked real FRs).
- Without patient 06, the calibration changes a lot (rate 1.1 versus 2.5 per minute): one
  patient dominates.

### Exercise
Replace the Hann-windowed sinusoids by real FR waveforms extracted from the training patients,
and check whether the ML detector trained on hybrid data improves, without leakage.

---

## Step 5 – Clinical analysis

### What we did and why
For each patient, the post-resection FR rate (per channel-minute), computed with the reference
markings and with each detector. Each detector was tuned without that patient, on
pre-resection FR markings only; the outcome is never used for tuning. The rate is compared with
the outcome (not seizure-free = positive).

### Key concepts
- **AUC** = P(score of a random positive > score of a random negative) = U / (n₊·n₋).
- **Stratified bootstrap**: resample within each class.
- **Permutation p-value**: shuffle the outcome labels.
- **Wilson intervals** for proportions.

### Key code (`src/epidetect/clinical.py`)
```python
ranks = rankdata(scores)                         # average ranks for ties
u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2
auc = u / (n_pos * n_neg)
```

### Results
| Score | AUC [95 % CI] | Permutation p |
|---|---|---|
| Reference markings | 0.83 [0.62, 1.00] | 0.004 |
| Wavelet detector | 0.61 [0.36, 0.84] | 0.41 |
| Energy detector | 0.45 [0.21, 0.70] | 0.70 |
| ML detector | 0.40 [0.18, 0.64] | 0.45 |

- The rule "at least one reference FR after resection" has sensitivity 6/8 (0.75) and
  specificity 10/14 (0.71).
- The reference reproduces the original publication. Our detectors do not, because their false
  positives in seizure-free patients drown the signal.
- The channel-level analysis is implemented but skipped (no labels).

### Common mistakes
- A plain bootstrap with 8 positives, which gives undefined replicates.
- Reading an association as causation.
- Letting the outcome influence the detector tuning.

### Exercise
Recompute the AUC of the reference rate by hand from the 8 × 14 pairs in
`results/clinical_patients.csv`.

---

## Glossary

- **HFO** (high-frequency oscillation): a brief oscillation above about 80 Hz, with at least
  about 4 cycles above the background.
- **Ripple**: an HFO between 80 and 250 Hz.
- **Fast ripple (FR)**: an HFO between 250 and 500 Hz; a candidate marker of epileptogenic
  tissue.
- **Spike (interictal epileptic spike)**: a sharp transient of tens of milliseconds and hundreds
  of µV. After a band-pass filter, it can mimic an HFO.
- **SEEG**: stereo-EEG, depth electrodes implanted in the brain.
- **ECoG**: electrocorticography, grids or strips on the cortical surface (here
  intraoperative).
- **Bipolar montage**: each channel is the difference between two neighbouring contacts.
- **BIDS**: Brain Imaging Data Structure, a standard file organisation for neuroimaging and
  electrophysiology data.
- **NIX**: a neuroscience data model on top of HDF5, the format of the raw files.
- **SNR**: signal-to-noise ratio. Here, the mean power of an event divided by the background
  power in its band, in dB.
- **1/f background**: brain background whose power decreases as 1/f^β.
- **Amplifier floor**: the white electronic noise visible above about 260 Hz.
- **Envelope (Hilbert)**: the instantaneous amplitude |x + i·H{x}| of a narrow-band signal.
- **Line length**: Σ|x[n] − x[n−1]|, the length of the curve; sensitive to amplitude and
  frequency.
- **Morlet wavelet**: a complex sinusoid in a Gaussian window with a fixed number of cycles; a
  constant-Q band-pass.
- **Whitening**: normalising each frequency of a time–frequency map by its background level.
- **filtfilt**: forward–backward filtering, giving zero phase and a squared gain.
- **Gibbs phenomenon**: oscillations caused by truncating a spectrum (or a series), seen here as
  ringing around spikes.
- **Robust z-score / MAD**: (x − median) / (1.4826 × median absolute deviation).
- **Precision / recall / F1**: TP/(TP+FP), TP/(TP+FN), 2TP/(2TP+FP+FN).
- **Micro / macro average**: pool the counts, or average per-unit metrics.
- **Leave-one-patient-out (LOPO)**: each patient is tested once with everything tuned on the
  others.
- **GroupKFold**: cross-validation whose folds never split a group (patient).
- **Bootstrap**: resampling units with replacement to estimate a confidence interval.
- **Stratified bootstrap**: resampling within each class separately.
- **Permutation test**: shuffling labels to build the null distribution of a statistic.
- **AUC**: area under the ROC curve; the probability that a random positive scores above a
  random negative.
- **ILAE outcome**: surgical outcome scale; 1 = seizure-free.
- **Hybrid simulation**: synthetic events inserted into real background.
- **Domain shift**: training data that differ from the data the model is applied to.
- **Label noise**: wrong labels in training data, here unmarked real FRs labelled as negatives.
- **NPE** (neural posterior estimation): a simulation-based inference method that trains a
  neural network to output the posterior distribution of simulator parameters given data
  (`sbi` library); the optional step 7.
