# Data notes: intraoperative ECoG with fast-ripple markings

Written during step 0, **before** any loading code, from a read-only inspection of the raw files.
All counts below come from [`scripts/00_inventory_raw.py`](../scripts/00_inventory_raw.py); the
full per-patient table is in [`data_inventory.md`](data_inventory.md) (regenerate it with
`.\run.ps1 inventory`).

## 1. Source, licence, citation

- Repository: <https://gin.g-node.org/USZ_NCH/Intraoperative_ECoG_HFO> (Neurosurgery department,
  University Hospital Zurich).
- Licence: **CC BY-SA 4.0**. Attribution is required, and derived *data* (e.g. our BIDS
  conversion) must keep the same licence. That is one more reason not to redistribute it: we
  publish only the code that rebuilds it.
- Primary citation (stored in each file's metadata): Boran E. et al., "High-density ECoG improves
  the detection of high frequency oscillations that predict seizure outcome", *Clinical
  Neurophysiology* 130(10):1882–1888, 2019, doi:10.1016/j.clinph.2019.07.008.
- Download: the 22 files `Data_Patient_01.h5` … `Data_Patient_22.h5` (1.9 GB in total) go into
  `data/raw/`.

## 2. File format

Each file is **HDF5** written with the **NIX** data model (root attribute `format = "nix"`,
version 1.1.0). NIX is a neuroscience schema layered on HDF5: it stores *blocks* (here, recording
sessions), *data arrays* with explicit *dimensions*, *tags* (annotated time intervals) and a
*metadata* tree. The files can be read either with plain `h5py`, which we used for inspection, or
with the official `nixio` library.

```
/metadata/General/...                      institution, recording device, electrode type, publication
/metadata/Patient/properties/...           Age, Gender, Etiology, Resection area, Follow-up period, Outcome ILAE
/data/Data_Pre_Resection_Bipolar_Montage/
    data_arrays/ECoG_Bipolar_Channels/
        data                               float64, shape (n_channels, n_samples), unit "uV"
        dimensions/1/labels                bipolar channel names, e.g. b"01-02"
        dimensions/2                       sampling_interval = 0.0005 s, offset 0, unit "s"
    groups/FR markings/tags/<uuid>/        one tag per fast ripple (also mirrored in .../tags/)
        position = [channel_index, onset_s]
        extent   = [1, duration_s]
        units    = ["s"]
/data/Data_Post_Resection_Bipolar_Montage/  same layout (FR group is empty for most patients)
```

Metadata quirk: some string properties (`Gender`, `Etiology`, `Electrode type`) are stored
**character by character**, one NIX record per letter. The reader has to join them back into
a string.

## 3. Signals

| Property | Value |
|---|---|
| Sampling rate | **2000 Hz** for every block, so the Nyquist frequency is 1000 Hz |
| Acquisition | Nicolet C-Series amplifier, 16-bit ADC, hardware passband **1–800 Hz** (from metadata) |
| Unit | µV, stored as float64 |
| Montage | **already bipolar** (neighbouring contacts subtracted, e.g. `01-02`). The monopolar signals are **not** provided |
| Electrodes | AdTech subdural grids (4×8, 2×14, 2×8) and strips (1×4, 1×6, 1×8) |
| Recordings | 22 patients, one pre-resection and one post-resection block each. Pre-resection: 246 channels, 82.9 min in total. Post-resection: 195 channels, 100.2 min in total. Blocks last 1 to 12.5 min |

Implications:
- The fast-ripple band (250–500 Hz) lies well below both the hardware low-pass (800 Hz) and
  Nyquist (1000 Hz). The ripple band (80–250 Hz) is also fully available.
- Because the montage is fixed to bipolar, we cannot re-reference the data (no common-average
  reference).
- Channel labels are **not uniform** across patients: `01-02` (patients 01–11, 14) versus `1-2`
  (12, 13, 15–22). Patient 20 has an unusual montage `1-4, 4-2, 2-3`. Some channels are missing
  (e.g. `08-09` does not exist because 8 and 9 sit at the ends of two grid rows; others were
  presumably excluded as bad). The pre- and post-resection channel sets differ, because the grid is
  repositioned after the resection.
- Clinically irrelevant metadata (age, gender, etiology) are present but will not be used as model
  inputs.

## 4. Fast-ripple (FR) markings

- **How they were made:** the GIN README states that FRs were "detected using a previously
  validated automatic detector" and then **visually validated**. This is **not** an independent
  expert marking from scratch. The reference is therefore biased toward events that this
  detector finds. A detector that resembles it will look better than it really is, and true FRs
  missed by that detector are counted as our false positives. This must be stated in the README
  limitations.
- **Only fast ripples are marked.** There are **no ripple (80–250 Hz) markings**. Per-class
  evaluation of ripples is therefore possible **only on simulations**.
- **Channel index convention:** `position[0]` is **1-based** (MATLAB style), even though NIX
  indices are normally 0-based. We checked this on all 2,479 markings in two ways:
  1. The channel label in the tag name, e.g. `FR_153_Channel_22_25-26`, equals
     `labels[index - 1]` in 100% of cases and `labels[index]` in 0% of cases.
  2. The peak 250–500 Hz amplitude inside the marked window, relative to that channel's median,
     is consistently higher on channel `index - 1` than on channel `index`.

  The loader (step 1) will assert check 1 for every file.
- Onsets are in seconds from the start of the block, and durations range from 9.5 to 63.5 ms
  (per-patient medians: 9.5–19.5 ms). All markings fall inside the recording.
- Counts: **2,378 FRs pre-resection** (16 of 22 patients have at least one) and **101 FRs
  post-resection** (10 patients).
- **Strong imbalance between patients:** patient 06 alone has 1,413 of the 2,378 pre-resection
  FRs (≈59%). Six patients (12, 15, 16, 17, 21, 22) have **zero** pre-resection FRs, and several
  have only 1–6. Consequences:
  - Pooled metrics would mostly measure performance on patient 06. We will report per-patient
    metrics and macro-averages, and bootstrap **over patients**, not over events.
  - In leave-one-patient-out, a test patient with zero reference FRs has undefined recall. Only
    false positives can be counted for them, so we will report the false-positive rate per minute
    for those patients.
- We cannot tell from the files whether every channel and the whole duration were reviewed,
  although the README implies it. We assume exhaustive marking and list this as an assumption.

## 5. Clinical information

| Field | Content |
|---|---|
| `Outcome ILAE` | ILAE surgical outcome class. Here: **14 patients ILAE 1** (seizure-free), 3 ILAE 3, 5 ILAE 5 |
| `Resection area` | **free text only**, e.g. "L medial parietal" or "R posterior cingulate gyrus, posterior hippocampus" |
| `Follow-up period` | months (12–43) |

**Important: no channel-level resection information.** Nothing in the files or the README says
which contacts lie over tissue that was resected. The README mentions only the "approximate
locations of recordings and the resected area". As a result, the analysis originally planned for
step 5 (HFO rate per channel versus resected zone, ROC/AUC per patient) **cannot be done honestly
with this dataset**.

Rejected proxy: treating channels that disappear from the post-resection recording as
"resected". This is not valid, because the grid is moved and channels can be dropped for
technical reasons.

What the data does support, which is also the analysis of the original publication: a
**patient-level** prediction of outcome from **post-resection FR occurrence or rate**, with
seizure-free versus not seizure-free as the binary label. The idea is that FRs remaining after
the resection indicate residual epileptogenic tissue. With 22 patients (8 not seizure-free), any
AUC will have a very wide confidence interval. We will report it with a bootstrap CI and
present it as exploratory.

## 6. BIDS conversion (done in step 1)

Produced by `scripts/01_convert_to_bids.py`. The table below was the plan written in step 0;
the notes after it record what was actually done.

| Item | BIDS mapping | Notes |
|---|---|---|
| Patient `NN` | `sub-NN` | |
| Pre/post block | `ses-preresection` / `ses-postresection`, `task-intraop` | Same surgery, but two distinct recording conditions |
| Signal | `ieeg/`, BrainVision format, channel type `ecog` | float32 BrainVision avoids the rescaling that 16-bit EDF would need |
| Reference | `iEEGReference = "bipolar"` in the sidecar | Channel names keep the bipolar pair, e.g. `01-02` |
| FR markings | `events.tsv` with `trial_type = FR` | BIDS `events.tsv` has no standard channel column. The channel will be kept in an extra column or encoded in the description. To be checked against what `mne-bids` writes |
| Age, sex, etiology, resection area, ILAE, follow-up | `participants.tsv` + `participants.json` | |
| Electrode coordinates | `electrodes.tsv` **not written** | No coordinates are provided |
| Licence, citation | `dataset_description.json` | CC BY-SA 4.0, Boran et al. 2019 |

What was actually done:

- **FR channel.** `mne-bids` 0.20 writes `events.tsv` **without** the channel of
  channel-specific annotations (checked on a toy example). We therefore rewrite `events.tsv` with
  an extra `channel` column, described in `events.json`. `epidetect.io.read_bids_recording`
  rebuilds channel-specific annotations from it. Recordings without FRs still get an empty
  `events.tsv`.
- **Sidecar fields.** `mne-bids` wrote `Manufacturer = "Brain Products"`, which is the name of
  the *file format* writer. It also wrote `iEEGReference = "n/a"`. Both are overwritten with
  the real amplifier (Natus Nicolet C-Series), `bipolar`, and the 1–800 Hz hardware filters. The
  hardware filters also go into `channels.tsv` (`low_cutoff`, `high_cutoff`).
- **Electrode coordinates.** `mne-bids` writes `electrodes.tsv` with `n/a` coordinates and a
  coordinate system of `Other`. This is the honest encoding of "unknown".
- **Verification.** Every recording is read back from BIDS and compared with the NIX source:
  same channels, same FRs, maximum absolute error 3.9e-4 uV caused by float32 storage
  (`results/bids_conversion.csv`).
- **Official BIDS validator.** Command: `npx bids-validator@1.15.0 data/bids` (Node.js).
  Result on the step-1 output: "This dataset appears to be BIDS compatible", with **0 errors
  and 0 warnings**. Summary: 488 files, 902 MB, 22 subjects, 2 sessions, modality iEEG, task
  `intraop`.

## 7. Findings of the step-1 quality control

Produced by `scripts/02_quality_control.py`. Outputs: `results/qc_*.csv`, `figures/qc_*.png`.

1. **Duplicated recordings in the source data.** The pre-resection recordings of patients
   **12, 15 and 17** are **byte-identical**: 4 channels each, 122.2 s each, same samples. These
   are three different patients (different ages, etiologies and resection areas), so at most
   one of the three recordings can be genuine, and we cannot tell which one. Their
   post-resection recordings are distinct. Handling:
   - the three pre-resection recordings stay in BIDS (faithful conversion), with every channel
     marked `bad` and the reason given;
   - they are listed in `[data] excluded_recordings` of `configs/default.toml` and excluded
     from every analysis;
   - the QC script fails if an exclusion is not backed by the duplicate check
     (`results/qc_duplicates.csv`).

   Cost: none of the three has FR markings.
2. **Channel flags.** With 3 recordings excluded, 429 channels remain. 18 of them are flagged
   (`results/qc_channels.csv`). Almost all are "250–500 Hz background outliers", i.e. a high
   noise floor in the fast-ripple band. The 250–500 Hz background is estimated with a
   *median* Welch PSD, so it is not driven by FRs: its robust z-score is uncorrelated with the
   channel's FR count. Relative criteria are used only for recordings with at least 8 channels,
   so strips only get the absolute checks (flat channel, repeated samples). Flagged channels
   carry 18 reference FRs, all post-resection (patients 06 and 08).
   - **Repeated samples (found in step 2).** In **P22 pre-resection, channel 3-4**, 50 % of
     consecutive samples are identical: every sample appears twice. The channel was sampled
     at 1000 Hz and duplicated to 2000 Hz (zero-order hold). Its effective Nyquist frequency
     is 500 Hz, so it cannot carry fast ripples. Every other channel has less than 0.5 %
     repeated samples. An absolute QC criterion (`max_repeated_fraction = 0.1`) now flags it.
     It was found because it produced absurd values (FR-band ratio ~10^4) during the spike
     calibration of step 2.
3. **Bad-channel policy for steps 4–5.** The main analysis uses **all** channels of
   non-excluded recordings, because the authors marked FRs on some flagged channels. A
   **sensitivity analysis** repeats it without flagged channels.
4. **Line noise.** The bipolar montage removes most of the 50 Hz component: median harmonic
   peaks are usually below 6 dB (`figures/qc_line_noise.png`). Exceptions:
   - P14 pre-resection has **only even** harmonics (100–400 Hz, up to 22 dB);
   - P20 post-resection has **only odd** harmonics (150–450 Hz, 12–15 dB)
     (`figures/qc_harmonics_examples.png`).

   Several harmonics fall inside the ripple and FR bands. Whether to notch-filter them is
   decided in step 3.
5. **Amplifier noise floor.** In many recordings the PSD becomes flat (white) above roughly
   150–250 Hz (`figures/qc_psd_pre.png`, `figures/qc_psd_post.png`). In the FR band, events
   must therefore stand out from electronic noise rather than from brain background. The
   simulator (step 2) must reproduce this.
6. **Aperiodic exponent.** On analysed channels, the 2–40 Hz exponent β has a median of 2.55
   (IQR 2.22–2.88). This guides the simulator's 1/f^β background.

## 8. Summary of consequences for the project

1. Real-data evaluation covers **fast ripples only**. Ripples are evaluated on simulations only.
2. The reference markings come from an automatic detector plus visual validation, so the
   "ground truth" is detector-biased.
3. The clinical analysis becomes **patient-level outcome prediction from post-resection FRs**
   instead of channel-level comparison with the resected zone. The code is structured so that
   a channel-level analysis can be added if channel-level resection labels become available.
4. Patient imbalance (patient 06 ≈ 59% of the FRs) requires per-patient metrics and a
   patient-level bootstrap.
5. The montage is bipolar only, sampled at 2000 Hz, with a 1–800 Hz hardware passband.
6. The pre-resection recordings of patients 12, 15 and 17 are duplicates and are excluded.
7. QC-flagged channels are kept in the main analysis, and a sensitivity analysis excludes them.
